# Box Training Methods
This repository contains code which accompanies the paper [Capacity and Bias of Learned Geometric Embeddings for Directed Graphs (Boratko et al. 2021)](https://proceedings.neurips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html).

This code includes implementations of many geometric embedding methods:
- Vector Similarity and Distance
- Bilinear Vector Model [(Nickel et al. 2011)](https://openreview.net/forum?id=H14QEiZ_WS)
- ComplEx Embeddings [(Trouillon et al. 2016)](https://arxiv.org/abs/1606.06357)
- Order Embeddings [(Vendrov et al. 2015)](https://arxiv.org/abs/1511.06361) and Probabilistic Order Embeddings [(Lai and Hockenmaier 2017)](https://aclanthology.org/E17-1068.pdf)
- Hyperbolic Embeddings, including:
  - "Lorentzian" - uses the squared Lorentzian distance on the Hyperboloid as in [(Law et al. 2019)](http://proceedings.mlr.press/v97/law19a.html), trains undirected but uses the asymmetric score function from [(Nickel and Kiela 2017)](https://proceedings.neurips.cc/paper/2017/file/59dfa2df42d9e3d41f5b02bfc32229dd-Paper.pdf) to determine edge direction at inference
  - "Lorentzian Score" - uses the asymmetric score above directly in training loss 
  - "Lorentzian Distance" - Hyperbolic model for directed graphs as described in section 2.3 of [(Boratko et al. 2021)](https://proceedings.neurips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html)
- Hyperbolic Entailment Cones [(Ganea et al. 2018)](https://arxiv.org/abs/1804.01882)
- Gumbel Box Embeddings [(Dasgupta et al. 2020)](https://arxiv.org/abs/2010.04831)
- t-Box model as described in section 3 of [(Boratko et al. 2021)](https://proceedings.neurips.cc/paper/2021/hash/88d25099b103efd638163ecb40a55589-Abstract.html)

It also provides a general-purpose pipeline to explore correlation between graph characteristics and models' learning capabilities.

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

### Multilabel Classification (non-BioASQ)

Example **train** command for `multilabel_classification` task, which includes the MLC datasets used in [Patel et al. 2022](https://par.nsf.gov/servlets/purl/10392233)
```
box_training_methods train --task multilabel_classification \
--data_path ./data/box-mlc-iclr-2022-data/expr_FUN/ \
--model_type hard_box --dim 8 --epochs 25 --negative_sampler hierarchical --hierarchical_negative_sampling_strategy exact
```

### BioASQ (English)

Example command for `bioasq` task (BioASQ Task A):
```
box_training_methods train --task bioasq \
--data_path ./data/mesh/allMeSH_2020.json \
--mesh_parent_child_mapping_path ./data/mesh/MeSH_parent_child_mapping_2020.txt \
--mesh_name_id_mapping_path ./data/mesh/MeSH_name_id_mapping_2020.txt \
--model_type tbox --dim 8 --epochs 25 --negative_sampler hierarchical --hierarchical_negative_sampling_strategy exact
```

Example **eval** command:
```
/usr/bin/env python scripts/box-training-methods eval \
--task bioasq \
--data_path ./data/mesh/ \
--mesh_parent_child_mapping_path ./data/mesh/MeSH_parent_child_mapping_2020.txt \
--mesh_name_id_mapping_path ./data/mesh/MeSH_name_id_mapping_2020.txt \
--bioasq_huggingface_encoder nlpie/bio-distilbert-uncased \
--instance_encoder_path /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/bioasq_models/embeddings.epoch-0.step-10.pt \
--box_model_path /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/bioasq_models/tbox.epoch-0.step-10.pt
```

### MESINESP2 (Spanish)

Example **train** command:
```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 /usr/bin/env python scripts/box-training-methods train \
--task bioasq \
--data_path ./data/bioasq/MESINESP2/ \
--bioasq_train_path ./data/bioasq/MESINESP2/Subtrack2-Clinical_Trials/Train/training_set_subtrack2.json \
--bioasq_dev_path ./data/bioasq/MESINESP2/Subtrack2-Clinical_Trials/Development/development_set_subtrack2.json \
--bioasq_test_path ./data/bioasq/MESINESP2/Subtrack2-Clinical_Trials/Test/test_set_subtrack2.json \
--bioasq_english False \
--mesh_parent_child_mapping_path ./data/bioasq/MESINESP2/DeCS2020.parent_child_mapping.txt \
--mesh_name_id_mapping_path ./data/bioasq/MESINESP2/DeCS2020.tsv \
--ancestors_cache_dir /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/bioasq/MESINESP2/cache/ancestors \
--negatives_cache_dir /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/bioasq/MESINESP2/cache/negatives \
--model_type tbox --dim 4 --log_batch_size 0 --epochs 25 \
--bioasq_huggingface_encoder microsoft/biogpt
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


