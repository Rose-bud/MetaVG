import math
import os
import sys
from typing import Iterable

import torch
from torch import nn, optim
from torch.func import functional_call

import util.misc as utils
from util import box_ops

import logging
import torch.distributed as dist
import time
import datetime
from tqdm import tqdm

from collections import OrderedDict
from copy import deepcopy


class data_prefetcher():
    def __init__(self, loader, device):
        self.length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            samples, targets = next(self.loader)
            self.next_img, self.next_mask = samples.decompose()
            self.next_target = targets
        except StopIteration:
            self.next_img = self.next_mask = self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.to(self.device, non_blocking=True)
            self.next_mask = self.next_mask.to(self.device, non_blocking=True)
            tensor_dict = self.next_target.tensor_dict
            self.next_target.tensor_dict = {k: tensor_dict[k].to(self.device, non_blocking=True) for k in tensor_dict}

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img, mask, target = self.next_img, self.next_mask, self.next_target
        self.preload()
        return img, mask, target

    def __next__(self):
        img, mask, target = self.next()
        if img == None:
            raise StopIteration
        return img, mask, target

    def __iter__(self):
        return self

    def __len__(self):
        return self.length


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'
    max_iter = len(data_loader)
    end = time.time()
    prefetcher = data_prefetcher(data_loader, device)
    inner_lr = 1e-4 if epoch < 60 else 1e-5
    base_model = model.module if hasattr(model, "module") else model
    trainable_params = tuple(param for param in base_model.parameters() if param.requires_grad)
    iteration = 0

    while True:
        support_img, support_mask, support_target = prefetcher.next()
        if support_img is None:
            break
        query_img, query_mask, query_target = prefetcher.next()
        if query_img is None:
            break

        support_targets = support_target.tensor_dict
        support_word_id = support_targets['word_id']
        support_word_mask = support_targets['word_mask']
        query_targets = query_target.tensor_dict
        query_word_id = query_targets['word_id']
        query_word_mask = query_targets['word_mask']
        data_time.update(time.time() - end)

        fast_weights = OrderedDict(base_model.named_parameters())
        fast_buffers = OrderedDict((name, buffer.clone()) for name, buffer in base_model.named_buffers())
        support_outputs = functional_call(
            base_model,
            (fast_weights, fast_buffers),
            (support_img, support_mask, support_word_id, support_word_mask),
        )
        support_loss_dict = criterion(support_outputs, support_targets)
        weight_dict = criterion.weight_dict
        support_loss = sum(
            support_loss_dict[key] * weight_dict[key]
            for key in support_loss_dict if key in weight_dict
        )
        adaptable_names = [name for name, param in fast_weights.items() if param.requires_grad]
        adaptable_params = [fast_weights[name] for name in adaptable_names]
        support_grads = torch.autograd.grad(
            support_loss, adaptable_params, create_graph=True, allow_unused=True
        )
        fast_weights = OrderedDict(fast_weights)
        for name, param, grad in zip(adaptable_names, adaptable_params, support_grads):
            if grad is not None:
                fast_weights[name] = param - inner_lr * (
                    grad / (grad.square().sqrt() + 1e-8) + 0.01 * param
                )

        query_outputs = functional_call(
            base_model,
            (fast_weights, fast_buffers),
            (query_img, query_mask, query_word_id, query_word_mask),
        )
        query_loss_dict = criterion(query_outputs, query_targets)
        query_loss = sum(
            query_loss_dict[key] * weight_dict[key]
            for key in query_loss_dict if key in weight_dict
        )
        loss_value = query_loss.detach().item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(query_loss_dict)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        meta_grads = torch.autograd.grad(query_loss, trainable_params, allow_unused=True)
        for param, grad in zip(trainable_params, meta_grads):
            if grad is not None:
                if utils.get_world_size() > 1:
                    dist.all_reduce(grad)
                    grad.div_(utils.get_world_size())
                param.grad = grad
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)
        optimizer.step()

        loss_dict_reduced = utils.reduce_dict(query_loss_dict)
        loss_dict_reduced_scaled = {
            key: value * weight_dict[key]
            for key, value in loss_dict_reduced.items() if key in weight_dict
        }
        loss_value = sum(loss_dict_reduced_scaled.values()).item()
        iteration += 1
        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 100 == 0 or iteration == max_iter // 2:
            eta_seconds = iter_time.global_avg * (
                max_iter // 2 - iteration + (max_iter // 2) * (epochs - epoch - 1)
            )
            eta_string = str(datetime.timedelta(seconds=max(0, int(eta_seconds))))
            logger.info(
                metric_logger.delimiter.join(
                    [
                        header,
                        "lr: {lr}",
                        "eta: {eta}",
                        "time: {time}",
                        "data: {data}",
                        "memory: {memory:.0f}",
                        "{meters}",
                    ]
                ).format(
                    epoch=epoch + 1,
                    iter=iteration,
                    max_iter=max_iter // 2,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger),
                )
            )

    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}

def train_one_epoch_lr(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()

    create_graph = False

    inner_lr = 0.000001

    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'


    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)

    img, mask, target = prefetcher.next()
    img1, mask1, target1 = img, mask, target
    img3, mask3, target3 = img, mask, target
    iteration = 0
    
    while img is not None:

        inner_opti = optim.Adam(model.parameters(), lr=1e-5)
        #inner_opti = optim.SGD(model.parameters(), lr=1e-10)
        weight0 = model.state_dict().copy()

        fast_weights = OrderedDict(model.named_parameters())
        # model.load_state_dict(fast_weights)
        fast_weights1 = fast_weights

        for i in range(5):
            
            target_dict = target.tensor_dict
            word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
            # iteration = iteration + 1
            data_time.update(time.time() - end)

            outputs = model(img, mask, word_id, word_mask)

            loss_dict = criterion(outputs, target_dict) 
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            # losses_value = torch.zeros_like(losses)
            # print(losses)
            for item in fast_weights.values():
                item.requires_grad = True

            gradients = torch.autograd.grad(losses, fast_weights.values(), create_graph=True, allow_unused=True)  

            for i, params in enumerate(model.parameters()):
                params.grad = gradients[i]

            inner_opti.step()
            # inner_opti.zero_grad()
            # model.parameters().grad.zero_()
            for i, params in enumerate(model.parameters()):
                if params.grad != None:
                    params.grad.zero_()
                else:
                    pass

           
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        img, mask, target = prefetcher.next()
        if img == None:
            img3, mask3, target3 = img, mask, target
            img, mask, target = img1, mask1, target1
           
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        iteration = iteration + 1
        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)

        loss_dict = criterion(outputs, target_dict)  

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        gradients = torch.autograd.grad(losses, model.parameters(),create_graph=True, allow_unused=True)

        model.load_state_dict(weight0)
        for i, params in enumerate(model.parameters()):
            params.grad = gradients[i]

        optimizer.step()

        for i, params in enumerate(model.parameters()):
            if params.grad != None:
                params.grad.zero_()
            else:
                pass

        # optimizer.zero_grad()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        if iteration % 100 == 0 or iteration == max_iter:
        # if iteration % 100 == 0 :
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs - epoch - 1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch + 1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))

        if img3==None:
            img=None

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch2(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()

    create_graph = False


    # train_model = deepcopy(model)
    inner_lr = 0.00001

    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'


    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)

    img, mask, target = prefetcher.next()
    img1, mask1, target1 = img, mask, target
    img3, mask3, target3 = img, mask, target
    iteration = 0

    while img is not None:
        if prefetcher.next():
            target_dict = target.tensor_dict
            word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
            iteration = iteration + 1
            data_time.update(time.time() - end)

            outputs = model(img, mask, word_id, word_mask)

            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            iter_time.update(time.time() - end)
            end = time.time()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

            if iteration % 100 == 0 or iteration == max_iter:
                eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs - epoch - 1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    metric_logger.delimiter.join(
                        [header,
                         "lr: {lr}",
                         "eta: {eta}",
                         "time: {time}",
                         "data: {data}",
                         "memory: {memory:.0f}",
                         "{meters}"
                         ]
                    ).format(
                        epoch=epoch + 1, iter=iteration, max_iter=max_iter,
                        lr=optimizer.param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))
            break

        else:
            fast_weights = OrderedDict(model.named_parameters())  
            # model.load_state_dict(fast_weights)
            fast_weights1 = fast_weights

            target_dict = target.tensor_dict
            word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
            iteration = iteration + 1
            data_time.update(time.time() - end)

            outputs = model(img, mask, word_id, word_mask)
            loss_dict = criterion(outputs, target_dict)  
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses_value = torch.zeros_like(losses, requires_grad=True)
            # print(losses)
            for item in fast_weights.values():
                item.requires_grad = True

            gradients = torch.autograd.grad(losses, fast_weights.values(), create_graph=True, allow_unused=True)  

            for (name, param), grad in zip(fast_weights.items(), gradients):
                if grad == None:
                    (name, param) = (name, param)
                else:
                    (name, param) = (name, param - inner_lr * grad)

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()

            img, mask, target = prefetcher.next()
            if img == None:
                img3, mask3, target3 = img, mask, target
                img, mask, target = img1, mask1, target1
            target_dict = target.tensor_dict
            word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
            # iteration = iteration + 1
            data_time.update(time.time() - end)

            weight = model.named_parameters()
            fast_weights = OrderedDict(fast_weights)

            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in fast_weights.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            outputs = model(img, mask, word_id, word_mask)

            loss_dict = criterion(outputs, target_dict)  

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            iter_time.update(time.time() - end)
            end = time.time()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

            
            if iteration % 100 == 0 or iteration == max_iter:
                eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs - epoch - 1))
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    metric_logger.delimiter.join(
                        [header,
                         "lr: {lr}",
                         "eta: {eta}",
                         "time: {time}",
                         "data: {data}",
                         "memory: {memory:.0f}",
                         "{meters}"
                         ]
                    ).format(
                        epoch=epoch + 1, iter=iteration, max_iter=max_iter,
                        lr=optimizer.param_groups[0]["lr"],
                        eta=eta_string,
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                        meters=str(metric_logger)
                    ))

            if img3 == None:
                img = None

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch3(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target = prefetcher.next()
    iteration = 0
    while img is not None:
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        iteration = iteration + 1
        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)

        loss_dict = criterion(outputs, target_dict)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 100 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs - epoch - 1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch + 1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))

        img, mask, target = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_w_accum(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target = prefetcher.next()
    iteration = 0
    
    while img is not None:
        target_dict = target.tensor_dict
        iteration = iteration + 1
        data_time.update(time.time() - end)

        B = img.shape[0]
        b = B // 2
        loss_dicts = list()
        weight_dict = criterion.weight_dict
        for i in range(2):
            b_img = img[i*b:(i+1)*b]
            b_mask = mask[i*b:(i+1)*b]
            b_target = {k: target_dict[k][i*b:(i+1)*b] for k in target_dict}
            b_word_id, b_word_mask = b_target['word_id'], b_target['word_mask']

            outputs = model(b_img, b_mask, b_word_id, b_word_mask)

            loss_dict = criterion(outputs, b_target)
            ##########################################
            ##########################################
            ##########################################
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) / 2
            losses.backward()
            loss_dicts.append(loss_dict)

        loss_dict_accum_scaled = {k: (loss_dicts[0][k] + loss_dicts[1][k]) * weight_dict[k] / 2
                                    for k in loss_dicts[0].keys() if k in weight_dict}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_scaled = utils.reduce_dict(loss_dict_accum_scaled)
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced_scaled)
            sys.exit(1)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 100 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch+1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))

        img, mask, target = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_epoch_eval(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader, device)
    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target_dict['orig_bbox']

        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)

        if criterion:
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        pred_boxes = postprocessor(outputs, target_dict)

        ious = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
        sum_iou = ious.sum()
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(img.size(0), device=img.device)

        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()

        all_pred_ious.append(ious.view(-1, 1))
        all_pred_boxes.append(pred_boxes)

    if save_path:
        torch.save({'pred_boxes': torch.cat(all_pred_boxes, dim=0),
                    'pred_ious': torch.cat(all_pred_ious, dim=0)},
                   save_path + 'pred_boxes')
    # accumulate predictions from all images
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample.float().item()
    miou = accum_iou.item() / accum_sample.float().item()

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    val_acc.update({'Mean_iou': miou})
    val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    return val_stats, val_acc, val_time

@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader, device)
    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target_dict['orig_bbox']

        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)

        if criterion:
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict


            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        pred_boxes = postprocessor(outputs, target_dict)

        ious = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
        sum_iou = ious.sum()
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(img.size(0), device=img.device)

        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()

        all_pred_ious.append(ious.view(-1, 1))
        all_pred_boxes.append(pred_boxes)

    if save_path:
        torch.save({'pred_boxes': torch.cat(all_pred_boxes, dim=0),
                    'pred_ious': torch.cat(all_pred_ious, dim=0)},
                   save_path + 'pred_boxes')
    # accumulate predictions from all images
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample.float().item()
    miou = accum_iou.item() / accum_sample.float().item()

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    val_acc.update({'Mean_iou': miou})
    val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    return val_stats, val_acc, val_time