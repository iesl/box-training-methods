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
from .loopers import GraphModelingEvalLooper
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

    if config["task"] == "graph_modeling":
        from box_training_methods.graph_modeling import train_eval as task_train_eval

    device = cuda_if_available(use_cuda=config["cuda"])

    if config["task"] == "graph_modeling":
        dataset = task_train_eval.setup_training_data(device, eval_only=True, **config)
        dataloader =  TensorDataLoader(dataset, batch_size=2 ** config["log_batch_size"], shuffle=False)
        model = task_train_eval.setup_model(dataset.num_nodes, device=device, eval_only=True, **config)

    # set Eval Looper
    eval_loopers = []
    if config["task"] == "graph_modeling":
        logger.debug(f"Will evaluate on full adjacency matrix")
        eval_loopers.append(
            GraphModelingEvalLooper(
                name="Eval",
                model=model,
                dl=dataloader,
                batchsize=2 ** config["log_eval_batch_size"],
                output_dir=config["output_dir"],
            )
        )

    if config["task"] == "graph_modeling":
        models= (model,)
    
    return models, eval_loopers
