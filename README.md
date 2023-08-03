## Installation

This repository makes use of submodules, to clone them you should use the `--recurse-submodules` flag, eg.
```bash
git clone <repo-url> --recurse-submodules
```
After cloning the repo, you should create an environment and install pytorch. For example,

```bash
conda create -n box-training-methods python=3.8
conda activate box-training-methods
conda install -c pytorch cudatoolkit=11.3 pytorch
```

You can then run `make all` to install the remaining modules and their dependencies. **Note:**
1. This will install Python modules, so you should run this command with the virtual environment created previously activated.
2. Certain graph generation methods (Kronecker and Price Network) will require additional dependencies to be compiled. In particular, Price requires that you use `conda`. If you are not interested in generating Kronecker or Price graphs you can skip this by using `make base` instead of `make all`.

## Usage

This module provides a command line interface available with `box_training_methods`.

### Graph Modeling

Example **training** command:
```
box_training_methods train --task graph_modeling \
--data_path ./data/graphs13/balanced_tree/branching\=10-log_num_nodes\=13-transitive_closure\=True/ \
--model_type tbox --dim 8 --epochs 25 --negative_sampler hierarchical --hierarchical_negative_sampling_strategy uniform
```

Example **eval** command (make sure the model hyperparams are the same as the ones the checkpoint was trained on):
```
/usr/bin/env python scripts/box-training-methods eval \
--data_path=data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/9.npz \
--task=graph_modeling \
--model_type=tbox --tbox_temperature_type=global --box_intersection_temp=0.01 --box_volume_temp=1.0 --log_eval_batch_size=17 --dim=128 \
--box_model_path /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/wandb/run-20230514_012634-zpbp23bk/files/learned_model.epoch-16.pt
```

## Citations
If you found the code contained in this repository helpful in your research, please cite the following papers:

```
@inproceedings{boratko2021capacity,
  title={Capacity and Bias of Learned Geometric Embeddings for Directed Graphs},
  author={Boratko, Michael and Zhang, Dongxu and Monath, Nicholas and Vilnis, Luke and Clarkson, Kenneth L and McCallum, Andrew},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
@article{patel2022modeling,
  title={Modeling label space interactions in multi-label classification using box embeddings},
  author={Patel, Dhruvesh and Dangati, Pavitra and Lee, Jay-Yoon and Boratko, Michael and McCallum, Andrew},
  journal={ICLR 2022 Poster},
  year={2022}
}
```


