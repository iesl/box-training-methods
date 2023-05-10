import json
import math
import os
import random
import uuid
from pathlib import Path
from time import time
from typing import *

import scipy.sparse
import toml
import torch
import wandb
from loguru import logger
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from wandb_utils.loggers import WandBLogger

from pytorch_utils import TensorDataLoader, cuda_if_available
from pytorch_utils.training import EarlyStopping, ModelCheckpoint
from .loopers import MultilabelClassificationEvalLooper
from box_training_methods import metric_logger


__all__ = [
    "evaluation",
    "setup",
]


def evaluation(config: Dict) -> None:
    """
    Setup and run evaluation loop.
    In this function we do any config manipulation required (eg. override values, set defaults, etc.)

    In practice we only need a separate eval procedure for BioASQ task, given its size
    
    :param config: config dictionary
    :return: None
    """

    models, eval_loopers = setup(**config)

    if config["wandb"]:
        metric_logger.metric_logger = WandBLogger()
        wandb.watch(models)
        for eval_looper in eval_loopers:
            eval_looper.summary_func = wandb.run.summary.update

    for eval_looper in eval_loopers:
        eval_looper.logger = metric_logger.metric_logger

    for eval_looper in eval_loopers:
        eval_looper.loop()

    if config["wandb"]:
        wandb.finish()

    logger.info("Evaluation complete!")


def setup(**config):
    """
    Setup and return the datasets, dataloaders, model, and training loop required for training.

    :param config: config dictionary
    :return: Tuple of dataset collection, dataloader collection, model, and train looper
    """

    from box_training_methods.multilabel_classification import train_eval as task_train_eval

    device = cuda_if_available(use_cuda=config["cuda"])

    dev_dataset, test_dataset = task_train_eval.setup_mesh_training_data(device, eval_only=True, **config)
    dev_dataloader = DataLoader(dev_dataset, batch_size=2 ** config["log_batch_size"], collate_fn=dev_dataset.collate_mesh_fn)#, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=2 ** config["log_batch_size"], collate_fn=dev_dataset.collate_mesh_fn)#, num_workers=12)

    num_labels = len(dev_dataset.le.classes_)
    instance_dim = -1  # FIXME this doesn't matter for bioasq where dim is automatically determined
    box_model, instance_encoder = task_train_eval.setup_model(num_labels, instance_dim, device, eval_only=True, **config)

    # set Eval Looper
    eval_loopers = []
    eval_loopers.extend([
        MultilabelClassificationEvalLooper(
            name="Validation",
            box_model=box_model,
            instance_model=instance_encoder,
            dl=dev_dataloader,
            batchsize=2 ** config["log_batch_size"],
        ),
        MultilabelClassificationEvalLooper(
            name="Test",
            box_model=box_model,
            instance_model=instance_encoder,
            dl=test_dataloader,
            batchsize=2 ** config["log_batch_size"],
        )
    ])

    models = (box_model, instance_encoder)
    return models, eval_loopers
