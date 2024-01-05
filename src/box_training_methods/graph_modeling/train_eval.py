from __future__ import annotations

import time
from typing import *

import attr
import numpy as np
import networkx as nx
import torch
from loguru import logger
from scipy.sparse import coo_matrix
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange, tqdm

from pytorch_utils.exceptions import StopLoopingException
from pytorch_utils.loggers import Logger
from pytorch_utils.training import IntervalConditional
from ..metrics import *

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
from wandb_utils.loggers import WandBLogger

from box_training_methods.models.temps import (
    GlobalTemp,
    PerDimTemp,
    PerEntityTemp,
    PerEntityPerDimTemp,
)
from pytorch_utils import TensorDataLoader, cuda_if_available
from pytorch_utils.training import EarlyStopping, ModelCheckpoint
from .dataset import (
    edges_from_tsv,
    edges_and_num_nodes_from_npz,
    RandomNegativeEdges,
    HierarchyAwareNegativeEdges,
    GraphDataset,
)
from .loss import (
    BCEWithLogsNegativeSamplingLoss,
    BCEWithLogitsNegativeSamplingLoss,
    BCEWithDistancesNegativeSamplingLoss,
    MaxMarginOENegativeSamplingLoss,
    PushApartPullTogetherLoss,
)
from box_training_methods.models.box import BoxMinDeltaSoftplus, TBox
from box_training_methods.models.hyperbolic import (
    Lorentzian,
    LorentzianDistance,
    LorentzianScore,
    HyperbolicEntailmentCones,
)
from box_training_methods.models.poe import OE, POE
from box_training_methods.models.vector import VectorSim, VectorDist, BilinearVector, ComplexVector

__all__ = [
    "setup_model",
    "setup_training_data",
    "EvalLooper",
]


# TODO make num_nodes a kwarg
def setup_model(
    num_nodes: int, device: Union[str, torch.device], eval_only: bool = False, **config
) -> Tuple[Module, Callable]:
    # TODO: Break this out into the model directory
    model_type = config["model_type"].lower()
    if model_type == "gumbel_box":
        model = BoxMinDeltaSoftplus(
            num_nodes,
            config["dim"],
            volume_temp=config["box_volume_temp"],
            intersection_temp=config["box_intersection_temp"],
        )
        if not eval_only:
            loss_func = BCEWithLogsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "tbox":
        temp_type = {
            "global": GlobalTemp,
            "per_dim": PerDimTemp,
            "per_entity": PerEntityTemp,
            "per_entity_per_dim": PerEntityPerDimTemp,
        }
        Temp = temp_type[config["tbox_temperature_type"]]

        model = TBox(
            num_nodes,
            config["dim"],
            intersection_temp=Temp(
                config["box_intersection_temp"],
                0.0001,
                100,
                # 0.009,
                # 0.011,
                dim=config["dim"],
                num_entities=num_nodes,
            ),
            volume_temp=Temp(
                config["box_volume_temp"],
                0.01,
                1000,
                # 0.9,
                # 1.1,
                dim=config["dim"],
                num_entities=num_nodes,
            ),
        )
        if not eval_only:
            loss_func = BCEWithLogsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "hard_box":
        model = TBox(
            num_nodes,
            config["dim"],
            hard_box=True
        )
        if not eval_only:
            loss_func = PushApartPullTogetherLoss(config["negative_weight"])
    elif model_type == "order_embeddings":
        model = OE(num_nodes, config["dim"])
        if not eval_only:
            loss_func = MaxMarginOENegativeSamplingLoss(
                config["negative_weight"], config["margin"]
            )
    elif model_type == "partial_order_embeddings":
        model = POE(num_nodes, config["dim"])
        if not eval_only:
            loss_func = BCEWithLogsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "vector_sim":
        model = VectorSim(
            num_nodes,
            config["dim"],
            config["vector_separate_io"],
            config["vector_use_bias"],
        )
        if not eval_only:
            loss_func = BCEWithLogitsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "vector_dist":
        model = VectorDist(num_nodes, config["dim"], config["vector_separate_io"],)
        if not eval_only:
            loss_func = BCEWithDistancesNegativeSamplingLoss(
                config["negative_weight"], config["margin"],
            )
    elif model_type == "bilinear_vector":
        model = BilinearVector(
            num_nodes,
            config["dim"],
            config["vector_separate_io"],
            config["vector_use_bias"],
        )
        if not eval_only:
            loss_func = BCEWithLogitsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "complex_vector":
        model = ComplexVector(num_nodes, config["dim"],)
        if not eval_only:
            loss_func = BCEWithLogitsNegativeSamplingLoss(config["negative_weight"])
    elif model_type == "lorentzian":
        model = Lorentzian(
            num_nodes,
            config["dim"],
            config["lorentzian_alpha"],
            config["lorentzian_beta"],
        )
        if not eval_only:
            loss_func = BCEWithDistancesNegativeSamplingLoss(config["negative_weight"])
            # TODO: implement multi-dimensional scaling loss (cf. Max Law paper)
    elif model_type == "lorentzian_score":
        model = LorentzianScore(
            num_nodes,
            config["dim"],
            config["lorentzian_alpha"],
            config["lorentzian_beta"],
        )
        if not eval_only:
            loss_func = BCEWithLogitsNegativeSamplingLoss(config["negative_weight"])
            # TODO: implement multi-dimensional scaling loss (cf. Max Law paper)
    elif model_type == "lorentzian_distance":
        model = LorentzianDistance(
            num_nodes,
            config["dim"],
            config["lorentzian_alpha"],
            config["lorentzian_beta"],
        )
        if not eval_only:
            loss_func = BCEWithLogsNegativeSamplingLoss(config["negative_weight"])
            # TODO: implement multi-dimensional scaling loss (cf. Max Law paper)
    elif model_type == "hyperbolic_entailment_cones":
        model = HyperbolicEntailmentCones(
            num_nodes,
            config["dim"],
            config["hyperbolic_entailment_cones_relative_cone_aperture_scale"],
            config["hyperbolic_entailment_cones_eps_bound"],
        )
        if not eval_only:
            loss_func = MaxMarginOENegativeSamplingLoss(
                config["negative_weight"], config["margin"]
            )
    else:
        raise ValueError(f"Model type {config['model_type']} does not exist")
    model.to(device)

    if not eval_only:
        return model, loss_func
    
    model.load_state_dict(torch.load(config["box_model_path"]))

    return model


def setup_training_data(device: Union[str, torch.device], eval_only: bool = False, **config) -> GraphDataset:
    """
    Load the training data (either npz or tsv)

    :param device: device to load training data on
    :param config: config dictionary

    :returns: GraphDataset with appropriate negative sampling, ready for training
    """
    start = time()

    graph = Path(config["data_path"])
    if graph.is_dir() and not eval_only:
        graphs = list({file.stem for file in graph.glob("*.npz")})
        logger.info(f"Directory {graph} has {len(graphs)} graph files")
        selected_graph_name = random.choice(graphs)
        logger.info(f"Selected graph {selected_graph_name}")
    else:  # passing in a specific random seed
        selected_graph_name = graph.name[:-len(".npz")]
        graph = graph.parent
    config["data_path"] = str(graph / selected_graph_name)

    if config["undirected"] is None:
        config["undirected"] = config["model_type"] == "lorentzian"
        logger.debug(
            f"Setting undirected={config['undirected']} since model_type={config['model_type']}"
        )

    npz_file = Path(config["data_path"] + ".npz")
    tsv_file = Path(config["data_path"] + ".tsv")
    avoid_edges = None
    if npz_file.exists():
        training_edges, num_nodes = edges_and_num_nodes_from_npz(npz_file)
    elif tsv_file.exists():
        stats = toml.load(config["data_path"] + ".toml")
        num_nodes = stats["num_nodes"]
        training_edges = edges_from_tsv(tsv_file)
        avoid_file = Path(config["data_path"] + ".avoid.tsv")
        if avoid_file.exists():
            avoid_edges = edges_from_tsv(avoid_file)
            logger.debug(f"Loaded {len(avoid_edges)} edges to avoid from {avoid_file}")
    else:
        raise ValueError(
            f"Could not locate training file at {config['data_path']}{{.npz,.tsv}}"
        )

    if config["sample_positive_edges_from_tc_or_tr"].lower() == "tr":
        training_edges = torch.tensor(list(nx.transitive_reduction(nx.DiGraph(training_edges)).edges))
    elif config["sample_positive_edges_from_tc_or_tr"].lower() == "tc":
        training_edges = torch.tensor(list(nx.transitive_closure(nx.DiGraph(training_edges)).edges))
    training_edges = training_edges.to(device)

    if config["undirected"]:
        training_edges = torch.unique(torch.sort(training_edges, dim=-1).values, dim=0)
    if avoid_edges is None:
        diag = torch.arange(num_nodes, device=device)[:, None].expand(-1, 2)
        if config["undirected"]:
            # The following is not particularly memory efficient, but should serve our purpose
            avoid_edges = torch.cat((training_edges, training_edges[..., [1, 0]], diag))
            # TODO implement transitive closure of training_edges to be avoid_edges for undirected as well as directed
        else:
            training_edges_tc = torch.tensor(list(nx.transitive_closure(nx.DiGraph(training_edges)).edges), device=device)
            avoid_edges = torch.cat((training_edges_tc, diag))

    if not eval_only:
        if config["negative_sampler"] == "random":
            negative_sampler = RandomNegativeEdges(
                num_nodes=num_nodes,
                negative_ratio=config["negative_ratio"],
                avoid_edges=avoid_edges,
                device=device,
                permutation_option=config["negatives_permutation_option"],
            )
        elif config["negative_sampler"] == "hierarchical":
            negative_sampler = HierarchyAwareNegativeEdges(
                edges=training_edges,
                negative_ratio=config["negative_ratio"],
                cache_dir=os.path.dirname(npz_file),
                graph_name=selected_graph_name,
                load_from_cache=True
            )
        else:
            raise NotImplementedError
    else:
        negative_sampler = None

    dataset = GraphDataset(
        training_edges, num_nodes=num_nodes, negative_sampler=negative_sampler, graph_npz_file=npz_file
    )

    logger.info(f"Number of edges in dataset: {dataset.num_edges:,}")
    logger.info(f"Number of edges to avoid: {len(avoid_edges):,}")
    logger.info(
        f"Number of negative edges: {num_nodes * (num_nodes - 1) - dataset.num_edges:,}"
    )
    logger.info(f"Density: {100*dataset.num_edges / (num_nodes * (num_nodes -1)):5f}%")
    logger.debug(f"Total time spent loading data: {time()-start:0.1f} seconds")

    return dataset
