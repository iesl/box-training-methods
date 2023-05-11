import random
from time import time
from typing import *
import os
import toml
from pathlib import Path

import torch
from torch.nn import Module
from pytorch_utils import TensorDataLoader, cuda_if_available

from .dataset import edges_from_hierarchy_edge_list, ARFFReader, InstanceLabelsDataset, BioASQInstanceLabelsIterDataset
from box_training_methods.graph_modeling.dataset import RandomNegativeEdges, \
    HierarchicalNegativeEdges, GraphDataset

from box_training_methods.models.temps import (
    GlobalTemp,
    PerDimTemp,
    PerEntityTemp,
    PerEntityPerDimTemp,
)
from box_training_methods.models.box import BoxMinDeltaSoftplus, TBox
from box_training_methods.graph_modeling.loss import (
    BCEWithLogsNegativeSamplingLoss,
    BCEWithLogsNegativeSamplingLossMLC,
    BCEWithLogitsNegativeSamplingLoss,
    BCEWithDistancesNegativeSamplingLoss,
    MaxMarginOENegativeSamplingLoss,
    PushApartPullTogetherLoss,
)
from box_training_methods.multilabel_classification.instance_encoder import InstanceAsPointEncoder, InstanceAsBoxEncoder
from box_training_methods.multilabel_classification.instance_scorers.instance_as_box_scorers.hard_box_scorer import HardBoxScorer

from box_training_methods.multilabel_classification.bioasq_instance_encoder import BioASQInstanceEncoder

__all__ = [
    "setup_model",
    "setup_training_data",
    "EvalLooper"
]


def setup_model(num_labels: int, instance_dim: int, device: Union[str, torch.device], eval_only=False, **config) -> Tuple[Module, Callable]:
    temp_type = {
        "global": GlobalTemp,
        "per_dim": PerDimTemp,
        "per_entity": PerEntityTemp,
        "per_entity_per_dim": PerEntityPerDimTemp,
    }
    Temp = temp_type[config["tbox_temperature_type"]]

    box_model = TBox(
        num_labels,
        config["dim"],
        intersection_temp=Temp(
            config["box_intersection_temp"],
            0.0001,
            100,
            dim=config["dim"],
            num_entities=num_labels,
        ),
        volume_temp=Temp(
            config["box_volume_temp"],
            0.01,
            1000,
            dim=config["dim"],
            num_entities=num_labels,
        ),
    )
    if not eval_only:
        loss_func = BCEWithLogsNegativeSamplingLossMLC(config["negative_weight"])
    box_model.to(device)

    # TODO args from click
    if config["task"] == "multilabel_classification":
        instance_encoder = InstanceAsBoxEncoder(instance_dim=instance_dim, hidden_dim=64, output_dim=config["dim"])
    else:  # config["task"] == "bioasq"
        instance_encoder = BioASQInstanceEncoder(output_dim=config["dim"], huggingface_encoder=config["bioasq_huggingface_encoder"])
    instance_encoder.to(device)

    if not eval_only:
        return box_model, instance_encoder, loss_func

    instance_encoder.load_state_dict(torch.load(config["instance_encoder_path"]))
    box_model.load_state_dict(torch.load(config["box_model_path"]))

    return box_model, instance_encoder


def setup_training_data(device: Union[str, torch.device], **config) -> \
        Tuple[GraphDataset, InstanceLabelsDataset, InstanceLabelsDataset, InstanceLabelsDataset]:
    """
    Load the training data

    :param device: device to load training data on
    :param config: config dictionary

    :returns: MLCDataset ready for training
    """
    start = time()

    data_dir = Path(config["data_path"])
    assert data_dir.is_dir()
    hierarchy_edge_list_file = data_dir / "hierarchy.edgelist"

    # 1. read label taxonomy into GraphDataset
    taxonomy_edges, label_encoder = edges_from_hierarchy_edge_list(edge_file=hierarchy_edge_list_file,
                                                                   input_child_parent=True,
                                                                   output_child_parent=True)  # (child, parent) format for GraphDataset
    label_set = label_encoder.classes_
    num_labels = len(label_set)

    if config["negative_sampler"] == "random":
        negative_sampler = RandomNegativeEdges(
            num_nodes=num_labels,
            negative_ratio=config["negative_ratio"],
            avoid_edges=None,  # TODO understand the functionality in @mboratko's code
            device=device,
            permutation_option=config["negatives_permutation_option"],
        )
    elif config["negative_sampler"] == "hierarchical":
        negative_sampler = HierarchicalNegativeEdges(
            edges=taxonomy_edges,
            negative_ratio=config["negative_ratio"],
            sampling_strategy=config["hierarchical_negative_sampling_strategy"],
            # cache_dir=config["data_path"] + ".hns",
        )
    else:
        raise NotImplementedError

    taxonomy_dataset = GraphDataset(
        taxonomy_edges, num_nodes=num_labels, negative_sampler=negative_sampler
    )

    # 2. read instance-labels into InstanceLabelsDataset
    reader = ARFFReader(num_labels=num_labels)

    data_train = list(reader.read_internal(str(data_dir / "train-normalized.arff")))
    instance_feats_train = torch.tensor([i['x'] for i in data_train], device=device)
    labels_train = [i['labels'] for i in data_train]

    data_dev = list(reader.read_internal(str(data_dir / "dev-normalized.arff")))
    instance_feats_dev = torch.tensor([i['x'] for i in data_dev], device=device)
    labels_dev = [i['labels'] for i in data_dev]

    data_test = list(reader.read_internal(str(data_dir / "test-normalized.arff")))
    instance_feats_test = torch.tensor([i['x'] for i in data_test], device=device)
    labels_test = [i['labels'] for i in data_test]

    train_dataset = InstanceLabelsDataset(instance_feats=instance_feats_train, labels=labels_train, label_encoder=label_encoder)
    dev_dataset = InstanceLabelsDataset(instance_feats=instance_feats_dev, labels=labels_dev, label_encoder=label_encoder)
    test_dataset = InstanceLabelsDataset(instance_feats=instance_feats_test, labels=labels_test, label_encoder=label_encoder)

    # TODO update these stats
    # logger.info(f"Number of edges in dataset: {dataset.num_edges:,}")
    # logger.info(f"Number of edges to avoid: {len(avoid_edges):,}")
    # logger.info(
    #     f"Number of negative edges: {num_nodes * (num_nodes - 1) - dataset.num_edges:,}"
    # )
    # logger.info(f"Density: {100*dataset.num_edges / (num_nodes * (num_nodes -1)):5f}%")
    #
    # logger.info(f"Number of labels in dataset: {dataset.num_labels:,}")
    # logger.debug(f"Total time spent loading data: {time() - start:0.1f} seconds")
    #
    return taxonomy_dataset, train_dataset, dev_dataset, test_dataset


def setup_mesh_training_data(device: Union[str, torch.device], eval_only: bool = False, **config):

    bioasq_path = Path(config["data_path"])
    if "bioasq_train_path" in config:
        bioasq_train_path = Path(config["bioasq_train_path"])
    else:
        bioasq_train_path = bioasq_path / "train.jsonl"
    if "bioasq_dev_path" in config:
        bioasq_dev_path = Path(config["bioasq_dev_path"])
    else:
        bioasq_train_path = bioasq_path / "train.jsonl"
    if "bioasq_test_path" in config:
        bioasq_test_path = Path(config["bioasq_test_path"])
    else:
        bioasq_test_path = bioasq_path / "test.jsonl"

    mesh_parent_child_path = Path(config["mesh_parent_child_mapping_path"])
    mesh_name_id_path = Path(config["mesh_name_id_mapping_path"])

    if not eval_only:
        train_dataset = BioASQInstanceLabelsIterDataset(
            file_path=bioasq_train_path,
            parent_child_mapping_path=mesh_parent_child_path,
            name_id_mapping_path=mesh_name_id_path,
            ancestors_cache_dir=config["ancestors_cache_dir"],
            negatives_cache_dir=config["negatives_cache_dir"],
            huggingface_encoder=config["bioasq_huggingface_encoder"],
            train=True,
            english=config["bioasq_english"],
        )

    validation_dataset = BioASQInstanceLabelsIterDataset(
        file_path=bioasq_dev_path,
        parent_child_mapping_path=mesh_parent_child_path,
        name_id_mapping_path=mesh_name_id_path,
        ancestors_cache_dir=config["ancestors_cache_dir"],
        negatives_cache_dir=config["negatives_cache_dir"],
        huggingface_encoder=config["bioasq_huggingface_encoder"],
        train=False,
        english=config["bioasq_english"],
    )
    test_dataset = BioASQInstanceLabelsIterDataset(
        file_path=bioasq_test_path,
        parent_child_mapping_path=mesh_parent_child_path,
        name_id_mapping_path=mesh_name_id_path,
        ancestors_cache_dir=config["ancestors_cache_dir"],
        negatives_cache_dir=config["negatives_cache_dir"],
        huggingface_encoder=config["bioasq_huggingface_encoder"],
        train=False,
        english=config["bioasq_english"],
    )

    if not eval_only:
        return train_dataset, validation_dataset, test_dataset

    return validation_dataset, test_dataset
