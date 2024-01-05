import os
import math
import itertools
from pathlib import Path
from time import time
from collections import defaultdict
from typing import *

import attr
import numpy as np
import pandas as pd
import pickle
import torch
from loguru import logger
from scipy.sparse import load_npz, csr_matrix
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, WeightedRandomSampler

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

import networkx as nx
#from scipy.sparse._csr import csr_matrix

from box_training_methods.enums import PermutationOption

__all__ = [
    "node_counts",
    "parent_child_node_counts",
    "edges_from_tsv",
    "edges_from_pos_neg_hec_tsv",
    "edges_and_num_nodes_from_npz",
    "convert_edges_to_ints",
    "convert_ints_to_edges",
    "create_positive_edges_from_tails",
    "RandomEdges",
    "RandomNegativeEdges",
    "HierarchicalNegativeEdges",
    "HierarchyAwareNegativeEdges",
    "GraphDataset",
]


def node_counts(indices: LongTensor, num_nodes: int) -> LongTensor:
    """
    Count the number of times each node appears in indices.

    :param indices: LongTensor of node indices
    :param num_nodes: number of nodes
    :return: count vector `x` such that `x[i]` is the number of times `i` appears in `indices`
    """
    unique_indices, nonzero_counts = torch.unique(indices, return_counts=True)
    counts = torch.zeros(num_nodes, dtype=torch.long, device=indices.device)
    counts[unique_indices] = nonzero_counts
    return counts


def parent_child_node_counts(edges: LongTensor, num_nodes: int) -> LongTensor:
    """
    Separately count the number of times each node appears as a parent and as a child.

    :param edges: Edges represented as a LongTensor with shape (..., 2), where [...,0] is
        parent and [..., 1] is child.
    :param num_nodes: integer representing total number of nodes
    :return: LongTensor of shape (num_nodes, 2) where [i,0] is the number of times i appears
        as a parent and [i,1] the number of times i appears as a child
    """
    parent_count = node_counts(edges[..., 0], num_nodes)
    child_count = node_counts(edges[..., 1], num_nodes)
    return torch.stack((parent_count, child_count), dim=-1)


def edges_from_tsv(edge_file: Union[Path, str]) -> LongTensor:
    """
    Loads edges from a given tsv file into a PyTorch LongTensor.
    Meant for importing data where each edge appears as a line in the file, with
        <parent_id>\t<child_id>

    :param edge_file: Path of edge tsv file in format above
    :returns: PyTorch LongTensor of edges with shape (num_edges, 2)
    """
    start = time()
    logger.info(f"Loading edges from {edge_file}...")
    edges = torch.tensor(pd.read_csv(edge_file, sep="\t", header=None).to_numpy())
    logger.info(f"Loading complete, took {time() - start:0.1f} seconds")
    return edges


def edges_from_pos_neg_hec_tsv(
    pos_edge_path: Union[Path, str], neg_edge_path: Union[Path, str], num_neg: int = 10
) -> LongTensor:
    """
    Loads separate positive and negative tsv files into a single PyTorch LongTensor
    Meant for importing data in the format used by Hyperbolic Entailment Cones, where each line in the positive
    tsv file corresponds to 10 lines of the negatives file.

    :param pos_edge_path: Path of positive edge tsv file
    :param neg_edge_path: Path of positive edge tsv file
    :param num_neg: Number of negatives for each positive
        Note: this shouldn't be *decided* in this function, it is a property of the data. If this number is 10, for
        example, then lines 1-10 of neg_edge_path will be associated with the positive edge on line 1 of pos_edge_path.
    :returns: Pytorch LongTensor of edges with shape (num_pos_edges, 1 + neg_ratio, 2), where [i,0,:] is the positive
        edge and [i,1:,:] are the (associated, permuted) negatives
    """
    pos_edge_path = edges_from_tsv(pos_edge_path)[:, None, :]
    neg_edge_path = edges_from_tsv(neg_edge_path).view(-1, num_neg, 2)
    return torch.cat((pos_edge_path, neg_edge_path), dim=1)


def edges_and_num_nodes_from_npz(path: Union[str, Path]) -> Tuple[LongTensor, int]:
    """
    Loads edges and number of nodes from an npz file. Meant for importing synthetic graph data.

    :param path: Location of npz file.
    :returns: Pytorch LongTensor of edges and int representing number of nodes
    """
    start = time()
    logger.info(f"Loading {path}...")
    digraph_coo = load_npz(path)
    logger.info(f"Creating PyTorch LongTensor representation of edges...")
    out_node_list = digraph_coo.row
    in_node_list = digraph_coo.col
    edges = torch.from_numpy(np.stack((out_node_list, in_node_list), axis=-1)).long()
    num_nodes = digraph_coo.shape[0]
    logger.info(f"Loading complete, took {time() - start:0.1f} seconds")
    return edges, num_nodes


def _validate_edge_tensor(instance, attribute, value) -> None:
    """
    Validator that ensures edges tensor is a LongTensor with shape (..., 2)
    """
    if not isinstance(value, Tensor):
        raise ValueError(f"Edges should be a LongTensor, but received {type(value)}")
    if value.dtype != torch.long:
        raise ValueError(
            f"Edges should have dtype=torch.long, but received edges with dtype={value.dtype}"
        )
    if value.shape[-1] != 2:
        raise ValueError(
            f"Edges should have shape (..., 2), but received edges with shape {value.shape}"
        )


def _node_probability_converter(
    node_probabilities: Optional[Tensor],
) -> Optional[Tensor]:
    """
    Convert node probabilities to the proper shape and type for use with multinomial.

    :param node_probabilities: Tensor of shape (num_nodes) or (num_nodes, 2)
    :return: transposed and expanded form of node_probabilities with shape (2, num_nodes),
        which can be used with torch.multinomial
    """
    if node_probabilities is None:
        return None
    original_shape = node_probabilities.shape
    if len(node_probabilities.shape) == 1:
        node_probabilities = node_probabilities[:, None].expand(-1, 2)
    if len(node_probabilities.shape) != 2 or node_probabilities.shape[1] != 2:
        raise ValueError(
            f"node_probabilities.shape={original_shape} should have shape (num_nodes,) or (num_nodes, 2)"
        )
    return node_probabilities.transpose(0, 1).float()


def convert_edges_to_ints(edges: LongTensor, num_nodes: int) -> LongTensor:
    """
    Convert edges from (i, j) form to integers k = i*num_nodes + j.

    :param edges: LongTensor with shape (..., 2) where edges[...,0] is the tail and edges[...,1] is the head.
    :param num_nodes: number of nodes
    :returns: LongTensor with shape (...,) where the value is an integer representation of an edge
    """
    return edges[..., 0] * num_nodes + edges[..., 1]


def convert_ints_to_edges(ints: LongTensor, num_nodes: int) -> LongTensor:
    """
    Convert integers k to edges (k // num_nodes, k % num_nodes)

    :param ints: LongTensor with shape (...,) where the value is an integer representation of an edge
    :param num_nodes: number of nodes
    :returns: LongTensor with shape (..., 2) where [...,0] is the node id of tail and [...,1] is the node id of head.
    """
    return torch.stack(
        (torch.div(ints, num_nodes, rounding_mode="trunc"), ints % num_nodes), dim=-1
    )


# TODO figure out whether head or tail resides at position 0
def create_positive_edges_from_tails(tails: LongTensor, A: csr_matrix) -> LongTensor:
    """

    :param tails: indices of nodes with shape (batch_size,) whose parents (potentially with >1 multiplicity) must be found and put into edges
    :param A: sparse adjacency matrix
    :returns: LongTensor with shape (batch_size+, 2) where [...,0] is the node id of head and [...,1] is the node id of tail
    """
    heads, tail_idxs = A[:,tails].nonzero()
    heads, tail_idxs = LongTensor(heads), LongTensor(tail_idxs)
    tails = torch.gather(input=tails, dim=0, index=tail_idxs)
    heads_tails = torch.cat([heads.unsqueeze(-1), tails.unsqueeze(-1)], dim=-1)
    return heads_tails


@attr.s(auto_attribs=True)
class RandomEdges:
    """
    Return randomly sampled edges, using various probabilities.
    Note: These are simply random edges, not random *negative* edges. This class makes no attempt to verify that these
    edges do not appear in the graph.

    This uses no additional RAM, apart from any tensors provided as input.

    :param num_nodes: Number of nodes in the graph
    :param negative_ratio: Number of negatives to sample for each positive
    :param permuted: If True, for each positive edge (i,j) return negative edges (i', j) or (j, i').
    :param permuted_probabilities: If provided, is used to determine if (i,j) should be permuted to (i', j) or (j', i).
    :param node_probabilities: If provided, will sample new nodes according to this tensor of weights.
        Should have shape (num_nodes,) or (num_nodes, 2), where the latter indicates different weights for tail/head;
            node_probabilities[i,0] = probability to sample i as tail node
            node_probabilities[i,1] = probability to sample i as head node
    """

    num_nodes: int
    negative_ratio: int = 10
    permuted: bool = False
    permuted_probabilities: Optional[Tensor] = None
    node_probabilities: Optional[Tensor] = attr.ib(
        default=None, converter=_node_probability_converter
    )
    avoid_edges: Optional[LongTensor] = None

    def __call__(self, positive_edges: Optional[LongTensor]) -> LongTensor:
        """
        Return negative edges for each positive edge.

        :param positive_edges: Positive edges, a LongTensor of indices with shape (..., 2)
        :return: negative edges, a LongTensor of indices with shape (..., negative_ratio, 2)
        """

        negative_shape = (*positive_edges.shape[:-1], self.negative_ratio, 2)

        # sample negatives
        if self.node_probabilities is None:
            negative_edges = torch.randint(
                self.num_nodes, negative_shape, device=positive_edges.device
            )
        else:
            num_samples = math.prod(negative_shape[:-1])
            negative_edges = (
                torch.multinomial(
                    self.node_probabilities, num_samples, replacement=True,
                )
                .transpose(0, 1)
                .reshape(negative_shape)
                .to(positive_edges.device)
            )

        if self.permuted:
            if self.permuted_probabilities is None:
                permuted_probabilities = 0.5 * torch.ones(negative_shape[:-1])
            else:
                raise NotImplementedError(
                    "Using non-uniform permuted probabilities is not supported yet"
                )
            negative_node_mask = (torch.bernoulli(permuted_probabilities) == 1).to(
                negative_edges.device
            )
            # a 0 means (i',j), a 1 means (i,j')
            negative_node_mask = torch.stack(
                (negative_node_mask, ~negative_node_mask), dim=-1
            )
            negative_edges = negative_edges.where(
                negative_node_mask, positive_edges[..., None, :]
            )

        return negative_edges

    def to(self, device: Union[str, torch.device]):
        if self.permuted_probabilities is not None:
            self.permuted_probabilities = self.permuted_probabilities.to(device)
        if self.node_probabilities is not None:
            self.node_probabilities = self.node_probabilities.to(device)
        if self.avoid_edges is not None:
            self.avoid_edges = self.avoid_edges.to(device)
        return self


class RandomNegativeEdges:
    """
    Return randomly sampled (true) negative edges.

    If permutation_option is none, this class uses O(len(avoid_edges)) RAM.
    If permutation_option is head or tail, this class uses O(len(avoid_edges) + num_nodes) RAM.

    TODO: allow for permutation with uniform probability of switching head or tail.
        This can be accomplished using no more than twice the RAM of the current head or tail implementation.

    :param num_nodes: Number of nodes in the graph.
    :param negative_ratio: Number of negatives to sample for each positive.
    :param avoid_edges: If provided, avoid these edges when sampling.
        Should be a LongTensor of shape (..., 2)
    :param device: Specify the device the sampling should be done on.
    :param permutation_option: Specify whether to permute head or tail nodes.
        Default is to perform uniform negative sampling, i.e. PermutationOption.none
    """

    def __init__(
        self,
        num_nodes: int,
        negative_ratio: int,
        avoid_edges: Optional[LongTensor] = None,
        device: Union[None, str, torch.device] = None,
        permutation_option: PermutationOption = PermutationOption.none,
    ):
        self._num_nodes = num_nodes
        self._sample_max = self._num_nodes ** 2
        self.negative_ratio = negative_ratio
        self._permutation_option = PermutationOption(permutation_option)
        if device is not None:
            self._device = device
        else:
            if avoid_edges is None:
                self._device = torch.device("cpu")
            else:
                self._device = avoid_edges.device

        if self.permutation_option != PermutationOption.none:
            self._breakpoints = (
                torch.arange(num_nodes + 1, device=self.device) * num_nodes
            )

        # The following repeats some functionality from pytorch_utils.random.RandomIntsAvoid, might be worth refactoring
        # this to avoid duplication.

        if avoid_edges is None:
            self._buckets = None
        else:
            if (
                self.permutation_option == PermutationOption.head
                or self.permutation_option == PermutationOption.none
            ):
                avoid_ints = convert_edges_to_ints(avoid_edges, self._num_nodes)
            elif self.permutation_option == PermutationOption.tail:
                avoid_ints = convert_edges_to_ints(
                    avoid_edges[..., [1, 0]], self._num_nodes
                )
            else:
                raise ValueError(
                    f"permutation_option={self.permutation_option} not supported."
                )

            avoid_ints = torch.unique(avoid_ints)
            self._sample_max -= len(avoid_ints)  # total number of negatives

            self._buckets = avoid_ints - torch.arange(len(avoid_ints), device=device)
            if self.permutation_option != PermutationOption.none:
                self._breakpoints = self._breakpoints - torch.bucketize(
                    self._breakpoints, avoid_ints, right=False
                )

    def __call__(self, positive_edges: Optional[LongTensor]) -> LongTensor:
        """
        Return negative edges for each positive edge.

        :param positive_edges: Positive edges, a LongTensor of indices with shape (..., 2), where [...,0] is the tail
            node index and [...,1] is the head node index.
        :return: negative edges, a LongTensor of indices with shape (..., negative_ratio, 2)
        """
        negative_int_shape = (*positive_edges.shape[:-1], self.negative_ratio)
        if self.permutation_option == PermutationOption.none:
            sample_idxs = torch.randint(
                self._sample_max, size=negative_int_shape, device=self.device
            )

        else:
            if self.permutation_option == PermutationOption.head:
                fixed_nodes = positive_edges[..., 0]
            elif self.permutation_option == PermutationOption.tail:
                fixed_nodes = positive_edges[..., 1]

            num_neg_per_node = (
                self._breakpoints[fixed_nodes + 1] - self._breakpoints[fixed_nodes]
            )
            sample_idxs = (
                torch.rand(negative_int_shape, device=self.device)
                * num_neg_per_node[..., None]
            ).long() + self._breakpoints[fixed_nodes][..., None]

        if self._buckets is not None:
            sample_idxs += torch.bucketize(sample_idxs, self._buckets, right=True)

        sample_edges = convert_ints_to_edges(sample_idxs, self._num_nodes)
        if self.permutation_option == PermutationOption.tail:
            # In this case, sample_idxs is interpreted as representing an edge (i,j) as j * num_nodes + i.
            # This is the reverse of our normal convention, so we need to swap head and tail.
            sample_edges = sample_edges[..., [1, 0]]
        return sample_edges

    @property
    def device(self):
        return self._device

    @property
    def permutation_option(self):
        return self._permutation_option

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self._random_ints_avoid.to(device)
        self._buckets.to(device)
        self._breakpoints.to(device)
        return self

'''
@attr.s(auto_attribs=True)
class HierarchyAwareNegativeEdges:

    edges: Tensor = attr.ib(validator=_validate_edge_tensor)
    aggressive_pruning: bool = False
    negative_ratio: int = 16
    cache_dir: str = ""
    graph_name: str = ""
    load_from_cache: bool = False

    def __attrs_post_init__(self):

        self._device = self.edges.device

        self.G = nx.DiGraph()
        self.G.add_edges_from((self.edges).tolist()) # assume nodes are numbered contiguously 0 through #nodes
        self.PAD = len(self.G.nodes)

        if self.load_from_cache:
            torch.load()
        
        else:
            self.hans_head2tails_dict = self.compute_hans_head2tails()
            self.hans_tail2heads_dict = self.invert_view(self.hans_head2tails_dict)  # ✓
            self.aggr_tail2heads_dict = self.compute_aggr_tail2heads()               # ✓
            
            # matrices for sampling, depending on hans/aggressive and tail-/head-oriented options
            self.hans_tail2heads_matrix = self.create_packed_padded_matrix_for_sampling(self.hans_tail2heads_dict)
            self.aggr_tail2heads_matrix = self.create_packed_padded_matrix_for_sampling(self.aggr_tail2heads_dict)

        if self.aggressive_pruning:
            self.tail2heads_matrix = self.aggr_tail2heads_matrix
        else:
            self.tail2heads_matrix = self.hans_tail2heads_matrix
        weights = torch.ones(shape=(len(self.G.nodes), 1), dtype=torch.float)
        weights = torch.vstack([weights, torch.tensor([1e-9])])
        self.weights = torch.nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=self.PAD)

    def __call__(self, positive_edges: Optional[LongTensor]) -> LongTensor:
        """
        Return negative edges for each positive edge.

        :param positive_edges: Positive edges, a LongTensor of indices with shape (..., 2), where [...,0] is the tail
            node index and [...,1] is the head node index.
        :return: negative edges, a LongTensor of indices with shape (..., negative_ratio, 2)
        """

        device = positive_edges.device

        tails = positive_edges[..., 0]
        negative_heads = self.tail2heads_matrix[tails].long().to(device)
        negative_heads_weights = self.weights.to(device)(negative_heads).squeeze()
        
        wrs = WeightedRandomSampler(weights=negative_heads_weights, num_samples=self.negative_ratio, replacement=True)
        wrs = list(wrs)
        negative_idxs = torch.tensor(wrs).to(device)
        try:
            negative_heads = torch.gather(negative_heads, -1, negative_idxs)
        except RuntimeError:
            # FIXME this happens when we have a leftover batch of one instance
            negative_heads = torch.gather(negative_heads, -1, negative_idxs.unsqueeze(dim=0))

        # FIXME for nodes with no HNS candidates, this will result in non-hierarchical negative_edges which may impact training
        #  fix this with masking
        negative_heads[negative_heads == self.PAD] = -1

        tails = tails.unsqueeze(-1).expand(-1, self.negative_ratio)
        negative_edges = torch.stack([tails, negative_heads], dim=-1)
        return negative_edges

    def compute_hans_head2tails(self):
        hans_head2tails_dict = defaultdict(list)
        for head in self.G.nodes:
            negative_tails_mres = self.hans_negative_tails_for_head(head)
            hans_head2tails_dict[head].extend(negative_tails_mres)
        return hans_head2tails_dict

    def hans_negative_tails_for_head(self, head):
        head_and_ancestors = {head} | nx.ancestors(self.G, head)
        negative_tails = set(self.G.nodes).difference(head_and_ancestors)
        G_negative_tails = nx.induced_subgraph(self.G, negative_tails)
        negative_tails_mres = [h for h in negative_tails if G_negative_tails.in_degree(h) == 0]
        return negative_tails_mres

    def compute_aggr_tail2heads(self):

        assert hasattr(self, "hans_tail2heads_dict")  # relies on previous computations

        aggr_tail2heads = defaultdict(list)
        for h, heads_h in self.hans_tail2heads_dict.items():
            G_heads_h = nx.induced_subgraph(self.G, heads_h)  # subgraph of G induced by heads corresponding to tail h in hans edges
            heads_h_star = [t for t in heads_h if G_heads_h.out_degree(t) == 0]  # keep only terminals of heads-induced subgraph
            aggr_tail2heads[h] = heads_h_star
        return aggr_tail2heads

    @staticmethod
    def invert_view(x_to_Y):
        y_to_X = {}
        for x,Y in x_to_Y.items():
            for y in Y:
                y_to_X.setdefault(y, []).append(x)
        return y_to_X

    def create_packed_padded_matrix_for_sampling(self, x_to_Y):
        sequences = [torch.tensor(x_to_Y.get(h, [self.PAD])) for h in sorted(self.G.nodes)]
        packed_sequence = pack_sequence(sequences, enforce_sorted=False)
        Y, _ = pad_packed_sequence(packed_sequence, batch_first=True, padding_value=self.PAD)
        return Y
    
    def cache(self):
        torch.save(self.hans_tail2heads_matrix,
                   os.path.join(self.cache_dir, self.graph_name + ".hans.pt"))
        torch.save(self.aggr_tail2heads_matrix,
                   os.path.join(self.cache_dir, self.graph_name + ".aggr.pt"))
    
    def load(self):
        self.hans_tail2heads_matrix = torch.load(os.path.join(self.cache_dir, self.graph_name + ".hans.pt"))
        self.aggr_tail2heads_matrix = torch.load(os.path.join(self.cache_dir, self.graph_name + ".aggr.pt"))

    @property
    def device(self):
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self.edges = self.edges.to(device)
        self.tail2heads_matrix = self.tail2heads_matrix.to(device)
        return self
'''

@attr.s(auto_attribs=True)
class HierarchyAwareNegativeEdges:

    edges: Tensor = attr.ib(validator=_validate_edge_tensor)
    negative_ratio: int = 16
    cache_dir: str = ""
    graph_name: str = ""
    load_from_cache: bool = False

    def __attrs_post_init__(self):

        self._device = self.edges.device

        # create graph
        self.G = nx.DiGraph()
        self.G.add_edges_from((self.edges).tolist())
        nodes = sorted(self.G.nodes)

        # create weights for sampling negative edges
        self.PAD = len(nodes)
        weights = torch.ones((len(nodes), 1), dtype=torch.float)
        weights = torch.vstack([weights, torch.tensor([1e-9])])
        self.weights = torch.nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=self.PAD)

        # load/create negative edges
        if self.load_from_cache:
            self.tail2heads_matrix = self.load_from_cache()
        else:
            G_tc = nx.transitive_closure(self.G)
            A_tc = nx.adjacency_matrix(G_tc, nodelist=nodes)
            Dec = [set(A_tc[[n], :].nonzero()[1]) for n in nodes]
            Anc = [set(A_tc[:, [n]].nonzero()[0]) for n in nodes]
            N = set(itertools.product(nodes, nodes)).difference({(n, n) for n in nodes})
            N.difference_update(set(G_tc.edges))
            for (u,y) in sorted(list(N.copy())):
                N.difference_update({(x,y) for x in Dec[u]})
                N.difference_update({(u,v) for v in Anc[y]})
            self.tail2heads_dict = defaultdict(list)
            for (t,h) in N:
                self.tail2heads_dict[t].append(h)
            self.tail2heads_matrix = self.create_packed_padded_matrix_for_sampling(self.tail2heads_dict)

    def __call__(self, positive_edges: Optional[LongTensor]) -> LongTensor:
        """
        Return negative edges for each positive edge.

        :param positive_edges: Positive edges, a LongTensor of indices with shape (..., 2), where [...,0] is the tail
            node index and [...,1] is the head node index.
        :return: negative edges, a LongTensor of indices with shape (..., negative_ratio, 2)
        """

        device = positive_edges.device

        tails = positive_edges[..., 0]
        negative_heads = self.tail2heads_matrix[tails].long().to(device)
        negative_heads_weights = self.weights.to(device)(negative_heads).squeeze()
        
        wrs = WeightedRandomSampler(weights=negative_heads_weights, num_samples=self.negative_ratio, replacement=True)
        wrs = list(wrs)
        negative_idxs = torch.tensor(wrs).to(device)
        try:
            negative_heads = torch.gather(negative_heads, -1, negative_idxs)
        except RuntimeError:
            # FIXME this happens when we have a leftover batch of one instance
            negative_heads = torch.gather(negative_heads, -1, negative_idxs.unsqueeze(dim=0))

        # FIXME for nodes with no negative candidates, this will result in non-hierarchical negative_edges which may impact training
        #  fix this with masking?
        negative_heads[negative_heads == self.PAD] = -1

        tails = tails.unsqueeze(-1).expand(-1, self.negative_ratio)
        negative_edges = torch.stack([tails, negative_heads], dim=-1)
        return negative_edges

    def create_packed_padded_matrix_for_sampling(self, x_to_Y):
        sequences = [torch.tensor(x_to_Y.get(x, [self.PAD])) for x in sorted(self.G.nodes)]
        packed_sequence = pack_sequence(sequences, enforce_sorted=False)
        Y, _ = pad_packed_sequence(packed_sequence, batch_first=True, padding_value=self.PAD)
        return Y
    
    def cache(self):
        torch.save(self.tail2heads_matrix, os.path.join(self.cache_dir, self.graph_name + ".neg.pt"))
    
    def load(self):
        self.tail2heads_matrix = torch.load(os.path.join(self.cache_dir, self.graph_name + ".neg.pt"))

    @property
    def device(self):
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self.edges = self.edges.to(device)
        self.tail2heads_matrix = self.tail2heads_matrix.to(device)
        return self


@attr.s(auto_attribs=True)
class GraphDataset(Dataset):
    """
    A map-style dataset, compatible with TensorDataloader.

    :param edges: LongTensor of ids of shape (num,...,2), where edges[...,0] is tail and edges[...,1] is head
        (Typically, shape is (num_edges, 2), but may also be (num_edges, 1 + num_negatives, 2) with positives located in (:, 0, :))
    :param negative_sampling: Callable which takes in a set of (positive) edges and returns negatives.
    """

    edges: Tensor = attr.ib(validator=_validate_edge_tensor)
    num_nodes: int
    negative_sampler: Optional[Callable[[LongTensor,], LongTensor]] = None
    graph_npz_file: Optional[str] = None

    def __attrs_post_init__(self):
        self._device = self.edges.device

    def __getitem__(self, idxs: LongTensor) -> LongTensor:
        """
        :param idxs: LongTensor of shape (...,) indicating the index of the positive edges to select
        :return: LongTensor of shape (..., 1 + num_negatives, 2) where the positives are located in [:,0,:]
        """
        edge_batch = self.edges[idxs]
        if self.negative_sampler is not None:
            negative_edge_batch = self.negative_sampler(edge_batch)
            edge_batch = torch.cat(
                (edge_batch[..., None, :], negative_edge_batch.to(self.device)), dim=-2
            )
        return edge_batch

    def __len__(self):
        return len(self.edges)

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def device(self):
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self.edges = self.edges.to(device)
        return self


if __name__ == "__main__":

    # edges = [(0,1), (0,2), (1,3), (1,4), (2,5), (2,6), (3,7), (4,7), (4,8), (5,8)]

    # G = nx.DiGraph()
    # G.add_edges_from(edges)
    # HNE = HierarchicalNegativeEdges(
    #     edges=torch.tensor(list(G.edges)), sampling_strategy="uniform"
    # )

    # G_tr = nx.transitive_reduction(G)
    # HNE_tr = HierarchicalNegativeEdges(
    #     edges=torch.tensor(list(G_tr.edges)), sampling_strategy="uniform"
    # )

    # G_tc = nx.transitive_reduction(G)
    # HNE_tc = HierarchicalNegativeEdges(
    #     edges=torch.tensor(list(G_tc.edges)), sampling_strategy="uniform"
    # )

    # assert torch.equal(HNE.negative_roots, HNE_tr.negative_roots)
    # assert torch.equal(HNE.negative_roots, HNE_tc.negative_roots)
    
    edges = [(0,2), (0,3), (1,5), (2,4), (4,5), (5,6)]
    # edges = [(0,1), (1,2), (2,3), (4,5), (5,6), (6,7)]
    G = nx.DiGraph()
    G.add_edges_from(edges)

    HANS = HierarchyAwareNegativeEdgesV2(edges=torch.tensor(list(G.edges)))
    
    breakpoint()