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
from .loopers import GraphModelingTrainLooper, MultilabelClassificationTrainLooper,\
    GraphModelingEvalLooper, MultilabelClassificationEvalLooper
from box_training_methods import metric_logger


__all__ = [
    "training",
    "setup",
]


def training(config: Dict) -> None:
    """
    Setup and run training loop.
    In this function we do any config manipulation required (eg. override values, set defaults, etc.)

    :param config: config dictionary
    :return: None
    """

    if config["wandb"]:
        # wandb.init(settings=wandb.Settings(start_method="fork"))
        wandb.init(settings=wandb.Settings(start_method="fork"),name=config.get("wandb_name", None), tags=config.get("wandb_tags", None))
        wandb.config.update(config, allow_val_change=True)
        config = wandb.config
        run_dir = Path(wandb.run.dir)
    else:
        run_dir = Path(".")

    if config["seed"] is None:
        if config["wandb"]:
            config.update({"seed": random.randint(0, 2 ** 32)}, allow_val_change=True)
        else:
            config["seed"] = random.randint(0, 2 ** 32)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    models, train_looper = setup(**config)

    if config["wandb"]:
        metric_logger.metric_logger = WandBLogger()
        wandb.watch(models)
        for eval_looper in train_looper.eval_loopers:
            eval_looper.summary_func = wandb.run.summary.update

    # TODO: refactor so train looper simply imports metric_logger
    train_looper.logger = metric_logger.metric_logger
    for eval_looper in train_looper.eval_loopers:
        eval_looper.logger = train_looper.logger

    logger.debug("Will save best model in RAM (but not on disk) for evaluation")
    if config["task"] == "graph_modeling":
        model_checkpoint = ModelCheckpoint(run_dir)
        train_looper.save_model = model_checkpoint
    elif config["task"] in {"multilabel_classification", "bioasq"}:
        box_model_checkpoint = ModelCheckpoint(run_dir, "learned_box_model.pt")
        instance_model_checkpoint = ModelCheckpoint(run_dir, "learned_instance_model.pt")
        train_looper.save_box_model = box_model_checkpoint
        train_looper.save_instance_model = instance_model_checkpoint

    # TODO standardize what the train_looper returns across tasks - what is predictions_coo?
    metrics, predictions_coo = train_looper.loop(config["epochs"])

    # saving output results
    if config["output_dir"] == None:
        output_parent_dir = Path(os.path.dirname(config["data_path"])) / "results"
    else:
        output_parent_dir = Path(config["output_dir"])
    model_string = config["model_type"]
    if model_string == "tbox":
        model_string += f"_{config['tbox_temperature_type']}"
    model_string += f"_{config['dim']}"
    output_dir = output_parent_dir / model_string
    output_dir.mkdir(parents=True, exist_ok=True)
    data_id = os.path.basename(config["data_path"])
    random_hex = wandb.run.id if config["wandb"] else uuid.uuid4().hex
    with open(output_dir / f"{data_id}_{random_hex}.metric", "w") as f:
        f.write(json.dumps(dict(config)))
        f.write("\n")
        f.write(json.dumps(metrics))

    if config["save_model"]:
        if config["task"] == "graph_modeling":
            model_checkpoint.save_to_disk(None)
        elif config["task"] in {"multilabel_classification", "bioasq"}:
            box_model_checkpoint.save_to_disk(None)
            instance_model_checkpoint.save_to_disk(None)

    # TODO standardize saving predictions for predictions from all tasks (graphs, labels, etc.)
    if config["save_prediction"]:
        if len(predictions_coo) > 0 and predictions_coo[0] is not None:
            filename_pred = f"{output_dir}/{graph_id}_{random_hex}.prediction"
            scipy.sparse.save_npz(filename_pred, predictions_coo[0])  # check this part
        else:
            raise ValueError(
                "save_prediction was requested, but no predictions returned from training loop"
            )

    if config["wandb"]:
        wandb.finish()

    logger.info("Training complete!")


def setup(**config):
    """
    Setup and return the datasets, dataloaders, model, and training loop required for training.

    :param config: config dictionary
    :return: Tuple of dataset collection, dataloader collection, model, and train looper
    """

    if config["task"] == "graph_modeling":
        from box_training_methods.graph_modeling import train_eval as task_train_eval
    elif config["task"] in {"multilabel_classification", "bioasq"}:
        from box_training_methods.multilabel_classification import train_eval as task_train_eval

    device = cuda_if_available(use_cuda=config["cuda"])

    # setup data
    # TODO task-specific setup_training_data
    if config["task"] == "graph_modeling":
        train_dataset = task_train_eval.setup_training_data(device, **config)
        train_dataloader = TensorDataLoader(
            train_dataset, batch_size=2 ** config["log_batch_size"], shuffle=True
        )
    elif config["task"] == "multilabel_classification":
        taxonomy_dataset, train_dataset, dev_dataset, test_dataset = task_train_eval.setup_training_data(device, **config)
        taxonomy_dataloader = TensorDataLoader(taxonomy_dataset, batch_size=2 ** config["log_batch_size"], shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_size=2 ** config["log_batch_size"], collate_fn=train_dataset.collate_fn, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=2 ** config["log_eval_batch_size"], collate_fn=dev_dataset.collate_fn, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=2 ** config["log_eval_batch_size"], collate_fn=test_dataset.collate_fn, shuffle=False)
    elif config["task"] == "bioasq":
        train_dataset, dev_dataset, test_dataset = task_train_eval.setup_bioasq_training_data(device, **config)
        train_dataloader = DataLoader(train_dataset, batch_size=2 ** config["log_batch_size"], collate_fn=train_dataset.collate_fn, num_workers=12)
        dev_dataloader = DataLoader(dev_dataset, batch_size=2 ** config["log_eval_batch_size"], collate_fn=dev_dataset.collate_fn, num_workers=12)
        test_dataloader = DataLoader(test_dataset, batch_size=2 ** config["log_eval_batch_size"], collate_fn=dev_dataset.collate_fn, num_workers=12)

    if isinstance(config["log_interval"], float):
        if config["task"] != "bioasq":
            config["log_interval"] = math.ceil(len(train_dataset) * config["log_interval"])
        else:
            config["log_interval"] = 10000
    logger.info(f"Log every {config['log_interval']:,} instances")
    logger.info(f"Stop after {config['patience']:,} logs show no improvement in loss")

    # setup model
    # TODO task-specific setup_model
    # TODO remove num_nodes explicit arg from setup_model API
    if config["task"] == "graph_modeling":
        model, loss_func = task_train_eval.setup_model(train_dataset.num_nodes, device, **config)
    elif config["task"] in {"multilabel_classification", "bioasq"}:
        if config["task"] == "multilabel_classification":
            num_labels = len(train_dataset.label_encoder.classes_)
            instance_dim = train_dataset.instance_dim
        else:
            num_labels = len(train_dataset.le.classes_)
            instance_dim = 768  # FIXME has to be same as encoder model output!
        box_model, instance_encoder, loss_func = \
            task_train_eval.setup_model(num_labels, instance_dim, device, **config)

    # setup optimizer
    if config["task"] == "graph_modeling":
        opt = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"], weight_decay=0.0
        )
    elif config["task"] in {"multilabel_classification", "bioasq"}:
        # TODO add more params to optimize
        opt = torch.optim.Adam(
            box_model.parameters(), lr=config["learning_rate"], weight_decay=0.0
        )

    # set Eval Looper
    eval_loopers = []
    if config["eval"]:
        if config["task"] == "graph_modeling":
            logger.debug(f"After training, will evaluate on full adjacency matrix")
            eval_loopers.append(
                GraphModelingEvalLooper(
                    # name="Train",  # this is used for logging to describe the dataset, which is the same data as in train
                    name="Eval",  # change this to have separate name for HNS experiments which log F1 at every epoch
                    model=model,
                    dl=train_dataloader,
                    batchsize=2 ** config["log_eval_batch_size"],
                    output_dir=config["output_dir"],
                )
            )
        elif config["task"] in {"multilabel_classification", "bioasq"}:
            eval_loopers.extend([
                MultilabelClassificationEvalLooper(
                    name="Validation",
                    box_model=box_model,
                    instance_model=instance_encoder,
                    dl=dev_dataloader,
                ),
                MultilabelClassificationEvalLooper(
                    name="Test",
                    box_model=box_model,
                    instance_model=instance_encoder,
                    dl=test_dataloader,
                )
            ])
    if config["task"] == "graph_modeling":
        train_looper = GraphModelingTrainLooper(
            name="Train",
            model=model,
            dl=train_dataloader,
            opt=opt,
            loss_func=loss_func,
            # exact_negative_sampling=True if (config["negative_sampler"] == "hierarchical" and
            #                                  config["hierarchical_negative_sampling_strategy"] == "exact") else False,
            exact_negative_sampling=False,
            eval_loopers=eval_loopers,
            log_interval=config["log_interval"],
            early_stopping=EarlyStopping("Loss", config["patience"]),
        )
    elif config["task"] in {"multilabel_classification", "bioasq"}:
        train_looper = MultilabelClassificationTrainLooper(
            name="Train",
            box_model=box_model,
            instance_model=instance_encoder,
            dl=train_dataloader,
            opt=opt,
            loss_func=loss_func,
            eval_loopers=eval_loopers,
            log_interval=config["log_interval"],
            early_stopping=EarlyStopping("Loss", config["patience"]),
            bioasq=config["task"]=="bioasq",
        )

    if config["task"] == "graph_modeling":
        models = (model,)
    elif config["task"] in {"multilabel_classification", "bioasq"}:
        models = (box_model, instance_encoder)

    return models, train_looper
