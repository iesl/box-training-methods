import math, random
from pathlib import Path
from time import time
import pickle
import json
import ijson
from itertools import cycle, islice
from typing import *

import attr
import numpy as np
import pandas as pd
import networkx as nx
import torch
from loguru import logger
from wcmatch import glob
from torch import Tensor, LongTensor
from torch.utils.data import Dataset, IterableDataset, DataLoader

from skmultilearn.dataset import load_from_arff
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer

__all__ = [
    "edges_from_hierarchy_edge_list",
    "name_id_mapping_from_file",
    "ARFFReader",
    "InstanceLabelsDataset",
    "collate_mesh_fn",
    "BioASQInstanceLabelsDataset",
]


def edges_from_hierarchy_edge_list(edge_file: Union[Path, str] = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/MeSH_parent_child_mapping_2020.txt", mesh=False) -> Tuple[LongTensor, LabelEncoder]:
    """
    Loads edges from a given tsv file into a PyTorch LongTensor.
    Meant for importing data where each edge appears as a line in the file, with
        <child_id>\t<parent_id>\t{}

    :param edge_file: Path of dataset's hierarchy{_tc}.edge_list
    :param mesh: implies <parent_id>\t<child_id>, as for "MeSH_parent_child_mapping_2020.txt"
    :returns: PyTorch LongTensor of edges with shape (num_edges, 2), LabelEncoder that numerized labels
    """
    start = time()
    logger.info(f"Loading edges from {edge_file}...")
    edges = pd.read_csv(edge_file, sep=" ", header=None).to_numpy()[:, :2]  # ignore line-final "{}"
    if mesh:
        edges[:, [0, 1]] = edges[:, [1, 0]]  # (parent, child) -> (child, parent)
    le = LabelEncoder()
    edges = torch.tensor(le.fit_transform(edges.flatten()).reshape((-1,2)))
    logger.info(f"Loading complete, took {time() - start:0.1f} seconds")
    return edges, le


def name_id_mapping_from_file(name_id_file: Union[Path, str] = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/MeSH_name_id_mapping_2020.txt") -> Dict:
    name_id = pd.read_csv(name_id_file, sep="=", header=None)
    name_id = dict(zip(name_id[0], name_id[1]))
    id_name = {i:n for n,i in name_id.items()}
    return name_id, id_name


# https://github.com/iesl/box-mlc-iclr-2022/blob/main/box_mlc/dataset_readers/arff_reader.py
class ARFFReader(object):
    """
    Reader for multilabel datasets in MULAN/WEKA/MEKA datasets.
    This reader supports reading multiple folds kept in separate files. This is done
    by taking in a glob pattern instread of single path.
    For example ::
            '.data/bibtex_stratified10folds_meka/Bibtex-fold@(1|2).arff'
        will match .data/bibtex_stratified10folds_meka/Bibtex-fold1.arff and  .data/bibtex_stratified10folds_meka/Bibtex-fold2.arff
    """

    def __init__(
        self,
        num_labels: int,
        labels_to_skip: Optional[List]=None,
        **kwargs: Any,
    ) -> None:
        """
        Arguments:
            num_labels: Total number of labels for the dataset.
                Make sure that this is correct. If this is incorrect, the code will not throw error but
                will have a silent bug.
            labels_to_skip: Some HMLC datasets remove the root nodes from the data. These can be specified here.
            **kwargs: Parent class args.
                `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py>`_
        """
        super().__init__(**kwargs)
        self.num_labels = num_labels
        if labels_to_skip is None:
            self.labels_to_skip = ['root', 'GO0003674', 'GO0005575', 'GO0008150'] # hardcode for HMLC datasets for now
            # we can hardcode more labels across datasets as long as they are to be skipped regardless of the dataset
            # because having a label name in this list that is not present in the dataset, won't affect anything.
        else:
            self.labels_to_skip = labels_to_skip

    def read_internal(self, file_path: str) -> List[Dict]:
        """Reads a datafile to produce instances
        Args:
            file_path: glob pattern for files containing folds to read
        Returns:
            List of json containing data examples
        """
        data = []

        for file_ in glob.glob(file_path, flags=glob.EXTGLOB | glob.BRACE):
            logger.info(f"Reading {file_}")
            x, y, feature_names, label_names = load_from_arff(
                file_,
                label_count=self.num_labels,
                return_attribute_definitions=True,
            )
            data += self._arff_dataset(
                x.toarray(), y.toarray(), feature_names, label_names
            )

        return data

    def _arff_dataset(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: List[Tuple[str, Any]],
        label_names: List[Tuple[str, Any]],
    ) -> List[Dict]:
        num_features = len(feature_names)
        assert x.shape[-1] == num_features
        num_total_labels = len(label_names)
        assert y.shape[-1] == num_total_labels
        all_labels = np.array([l_[0] for l_ in label_names])
        # remove root
        to_take = np.logical_not(np.in1d(all_labels, self.labels_to_skip)) # shape =(num_labels,),
        # where i = False iff we have to skip
        all_labels = all_labels[to_take]
        data = [
            {
                "x": xi.tolist(),
                "labels": (all_labels[yi[to_take] == 1]).tolist(),
                "idx": str(i),
            }
            for i, (xi, yi) in enumerate(zip(x, y))
            if any(yi)  # skip ex with empty label set
        ]

        return data


@attr.s(auto_attribs=True)
class InstanceLabelsDataset(Dataset):
    """
    """

    instance_feats: Tensor
    labels: Tensor
    label_encoder: LabelEncoder  # label set accessable via label_encoder.classes_

    def __attrs_post_init__(self):

        self._device = self.instance_feats.device
        self.instance_dim = self.instance_feats.shape[1]
        self.labels = self.prune_and_encode_labels_for_instances()
        
        instance_label_pairs = []
        for i, ls in enumerate(self.labels):
            instance_label_pairs.extend([i, l] for l in ls)
        self.instance_label_pairs = torch.tensor(instance_label_pairs)
        
        self.instance_feats = torch.nn.Embedding.from_pretrained(self.instance_feats, freeze=True)

    def __getitem__(self, idxs: LongTensor) -> LongTensor:
        """
        :param idxs: LongTensor of shape (...,) indicating the index of the examples which to select
        :return: instance_feats of shape (batch_size, instance_dim), label_idxs of shape (batch_size, num_labels)
        """
        instance_idxs = self.instance_label_pairs[idxs][:, 0].to(self._device)
        label_idxs = self.instance_label_pairs[idxs][:, 1].to(self._device)
        instance_feats = self.instance_feats(instance_idxs)
        return instance_feats, label_idxs

    def __len__(self):
        return len(self.labels)

    def prune_and_encode_labels_for_instances(self):
        pruned_labels = []
        for ls in self.labels:
            pruned_labels.append(self.label_encoder.transform(self.prune_labels_for_instance(ls)))
        return pruned_labels

    def prune_labels_for_instance(self, ls):
        """only retains most granular labels"""
        pruned_ls = []
        for i in range(len(ls)):
            label_i_is_nobodys_parent = True
            if i < len(ls) - 1:
                for j in range(i+1, len(ls)):
                    if f".{ls[j]}.".startswith(f".{ls[i]}."):
                        label_i_is_nobodys_parent = False
                        break
            if label_i_is_nobodys_parent:
                pruned_ls.append(ls[i])
        return pruned_ls

    @property
    def device(self):
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self.instance_feats = self.instance_feats.to(device)
        # self.labels = self.labels.to(device)
        return self


class MESHNegativeSampler(object):
    def __call__(self, labels: List[str]) -> List[str]:
        """Samples num_negatives labels from the mesh vocab that are not in labels"""
        # FIXME
        return []


def collate_mesh_fn(batch, tokenizer):
    inputs = tokenizer(
        [
            x["journal"]
            + f" {tokenizer.sep_token} "
            + x["title"]
            + f" {tokenizer.sep_token} "
            + x["abstractText"]
            for x in batch
        ],
        return_tensors="pt",
        padding=True,
    )
    # TODO: Handle variable number of positives and negatives using padding
    positives = torch.tensor(
        [[m for m in x["positives"]] for x in batch], dtype=torch.long
    )  # shape = (batch_size, num_positives)
    negatives = torch.tensor(
        [[m for m in x["negatives"]] for x in batch], dtype=torch.long
    )  # shape = (batch_size, num_negatives)
    extra_positive_edges = torch.tensor(
        [x["extra_positive_edges"] for x in batch], dtype=torch.long
    )  # shape = (batch_size, num_extra_positive_edges)
    return inputs, positives, negatives


@attr.s(auto_attribs=True)
class BioASQInstanceLabelsIterDataset(IterableDataset):
    """Dataset for BioASQ data with MESH labels
    Each data sample is a json object with the following keys:
    - journal: str
    - title: str
    - abstractText: str
    - meshMajor: List[str]
    where meshMajor is a list of MESH labels.
    """

    mesh_negative_sampler: MESHNegativeSampler
    file_path: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/allMeSH_2020.json"
    parent_child_mapping_path: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/MeSH_parent_child_mapping_2020.txt"
    name_id_mapping_path: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/MeSH_name_id_mapping_2020.txt"
    cycle: bool = True
    # TODO: DP: Wrap the dataset into a Shuffler instance to allow shuffling of the iterable dataset
    # https://pytorch.org/data/beta/generated/torchdata.datapipes.iter.Shuffler.html#torchdata.datapipes.iter.Shuffler

    def __attrs_post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        self.edges, self.le = edges_from_hierarchy_edge_list(
            edge_file=self.parent_child_mapping_path, mesh=True
        )
        self.name_id, self.id_name = name_id_mapping_from_file(
            name_id_file=self.name_id_mapping_path
        )
        self.G = nx.DiGraph(
            self.edges[:, [1, 0]].tolist()
        )  # reverse to parent-child format for DiGraph

    def label_to_id(self, labels: Iterable[str]) -> List[int]:
        return [self.name_id[label] for label in labels]

    def get_extra_positive_edges(
        self, labels: Iterable[str]
    ) -> Tuple[List[int], List[int]]:
        """Uses the MESH hierarchy to find extra positive edges for the labels.
        These are edges from positive children to positive ancestors.
        """
        # FIXME
        return []

    def parse_file(self, file_path):
        with open(file_path, encoding="windows-1252", mode="r") as f:
            for article in ijson.items(f, "articles.item"):
                article["positives"] = self.label_to_id(article["meshMajor"])
                article["extra_positive_edges"] = self.get_extra_positive_edges(
                    article["meshMajor"]
                )
                article["negatives"] = self.label_to_id(
                    self.mesh_negative_sampler(article["meshMajor"])
                )
                yield article

    def get_stream(self, file_path):
        if self.cycle:
            return cycle(self.parse_file(file_path))
        else:
            return self.parse_file(file_path)

    def __iter__(self):
        return self.get_stream(self.file_path)


def mesh_leaf_label_stats(bioasq: BioASQInstanceLabelsIterDataset):
    bioasq_iter = iter(bioasq)
    leaves = {n for n in bioasq.G.nodes if bioasq.G.out_degree(n) == 0}
    leaf_label_counts = []
    for i in range(10):
        label_names = next(bioasq_iter)['meshMajor']
        print(label_names)
        try:
            label_ids = [bioasq.name_id[l] for l in label_names]
        except KeyError:
            continue
        labels = bioasq.le.transform(label_ids)
        in_degrees = [bioasq.G.in_degree(l) for l in labels]
        out_degrees = [bioasq.G.out_degree(l) for l in labels]
        print("label in_degree:", in_degrees)
        print("label out_degree:", out_degrees)                        
        leaf_label_count = 0
        for l in labels:
            if l in leaves:
                leaf_label_count += 1
        leaf_label_counts.append(leaf_label_count)
        print(f"# leaf labels: {str(leaf_label_count)}")
    leaf_label_counts = np.array(leaf_label_counts)
    print(f"min: {str(np.min(leaf_label_counts))}")
    print(f"max: {str(np.max(leaf_label_counts))}")
    print(f"mean: {str(np.mean(leaf_label_counts))}")
    print(f"median: {str(np.median(leaf_label_counts))}")
    print(f"std: {str(np.std(leaf_label_counts))}")
    breakpoint()


def write_bioasq_pmids_to_file():
    bioasq = BioASQInstanceLabelsIterDataset(mesh_negative_sampler=MESHNegativeSampler(), cycle=False)
    bioasq_iter = iter(bioasq)
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/pmid_sorted_list.v3.txt", "+a") as f:
        while (x := next(bioasq_iter, None)) is not None:
            f.write(x['pmid'])
            f.write('\n')


def shuffle_and_split_bioasq_pmids():
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/pmid_sorted_list.v3.txt", "r") as f:
        pmids = [l.strip() for l in f.readlines() if l.strip()]
    random.shuffle(pmids)
    train, dev, test = pmids[:int(0.6 * len(pmids))], pmids[int(0.6 * len(pmids)): int(0.8 * len(pmids))], pmids[int(0.8 * len(pmids)):]
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/pmid.train.shuffled.txt", "w") as f:
        f.write("\n".join(train))
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/pmid.dev.shuffled.txt", "w") as f:
        f.write("\n".join(dev))
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/pmid.test.shuffled.txt", "w") as f:
        f.write("\n".join(test))


def distribute_mesh_articles_among_splits_based_on_pmids():

    bioasq = BioASQInstanceLabelsIterDataset(mesh_negative_sampler=MESHNegativeSampler(), cycle=False)
    bioasq_iter = iter(bioasq)
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/pmid.train.shuffled.txt", "r") as f:
        train_pmids = {l.strip() for l in f.readlines() if l.strip()}
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/pmid.dev.shuffled.txt", "r") as f:
        dev_pmids = {l.strip() for l in f.readlines() if l.strip()}
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/pmid.test.shuffled.txt", "r") as f:
        test_pmids = {l.strip() for l in f.readlines() if l.strip()}

    train_articles = []
    dev_articles = []
    test_articles = []
    while (x := next(bioasq_iter, None)) is not None:
        if x['pmid'] in train_pmids:
            train_articles.append(x)
        elif x['pmid'] in dev_pmids:
            dev_articles.append(x)
        elif x['pmid'] in test_pmids:
            test_articles.append(x)
        else:
            raise ValueError(f'pmid {pmid} not found!')

    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/train.2020.json", "a+") as f:
        f.write('{"articles":[')
        f.write('\n')
        for article in train_articles[:-1]:
            f.write(json.dumps(article))
            f.write(',\n')
        f.write(json.dumps(train_articles[-1]))
        f.write("]}")        
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/dev.2020.json", "a+") as f:
        f.write('{"articles":[')
        f.write('\n')
        for article in dev_articles[:-1]:
            f.write(json.dumps(article))
            f.write(',\n')
        f.write(json.dumps(dev_articles[-1]))
        f.write("]}")
    with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/test.2020.json", "a+") as f:
        f.write('{"articles":[')
        f.write('\n')
        for article in test_articles[:-1]:
            f.write(json.dumps(article))
            f.write(',\n')
        f.write(json.dumps(test_articles[-1]))
        f.write("]}")


if __name__ == "__main__":
    bioasq_test = BioASQInstanceLabelsIterDataset(mesh_negative_sampler=MESHNegativeSampler(), file_path="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/test.2020.json", cycle=False)
    bioasq_test_iter = iter(bioasq_test)
    print(next(bioasq_test_iter))
