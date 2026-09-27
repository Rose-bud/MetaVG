# MetaVG

This repository contains an implementation of **MetaVG: A Meta-Learning Framework for Visual Grounding**, published in *IEEE Signal Processing Letters* (2023). MetaVG is built on the VLTVG visual grounding codebase.

Paper: [IEEE Xplore](https://doi.org/10.1109/LSP.2023.3344374)

## Repository layout

Run the commands below from this directory (`MetaVG/MetaVG`). The default configuration is `configs/VLTVG_R50_referit.py`.

```text
configs/                 Experiment configuration
datasets/                Dataset loading and transforms
models/                  Model and visual grounding components
engine.py                Training and evaluation routines
train.py                 Training and validation entry point
test.py                  Legacy standalone evaluation script
data/                    Dataset images and annotations (prepare separately)
split/data/              Dataset split/index files
pretrained_checkpoints/  Pretrained model weights (prepare separately)
```


## Data, splits, and pretrained weights

Use the same datasets, split/index files, and pretrained model preparation as the [VLTVG README](https://github.com/yangli18/VLTVG/blob/master/README.md) and its [data setup guide](https://github.com/yangli18/VLTVG/blob/master/docs/get_started.md). Please follow that guide for downloading and arranging the data.

For the default ReferIt configuration, the training script expects the pretrained weights at:

```text
pretrained_checkpoints/detr-r50-referit.pth
```

The default dataset and index roots are `./data/` and `./split/data/`. If your data are stored elsewhere, pass `--data_root` and `--split_root` to the scripts.

## Training

The default run trains the ReferIt configuration:

```bash
python train.py --config configs/VLTVG_R50_referit.py
```


## Evaluation

Use the training entry point in evaluation mode to evaluate a saved checkpoint on the validation split:

```bash
python train.py --config configs/VLTVG_R50_referit.py --eval --resume output_dirs/VLTVG_R50_referit/checkpoint.pth
```

To evaluate the test split, add `--test_split test`:

```bash
python train.py --config configs/VLTVG_R50_referit.py --eval --test_split test --resume output_dirs/VLTVG_R50_referit/checkpoint.pth
```


## Reproducing other datasets

The training script accepts `--dataset`, `--data_root`, and `--split_root`. Use the dataset name and split files defined by VLTVG, and set the matching pretrained checkpoint and output directory. The included config is specifically set up for ReferIt; for another dataset, create or provide a configuration with the appropriate dataset name, pretrained weight path, and output directory.

## Citation

```bibtex
@article{su2023metavg,
  title={MetaVG: A meta-learning framework for visual grounding},
  author={Su, Chao and Li, Zhi and Lei, Tianyi and Peng, Dezhong and Wang, Xu},
  journal={IEEE Signal Processing Letters},
  volume={31},
  pages={236--240},
  year={2023},
  publisher={IEEE}
}
```
