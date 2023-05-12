import os
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

from box_training_methods.graph_modeling.dataset import RandomNegativeEdges, HierarchicalNegativeEdges

__all__ = [
    "edges_from_hierarchy_edge_list",
    "name_id_mapping_from_file",
    "ARFFReader",
    "InstanceLabelsDataset",
    "collate_mesh_fn",
    "BioASQInstanceLabelsDataset",
]


def edges_from_hierarchy_edge_list(edge_file: Union[Path, str] = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/MeSH_parent_child_mapping_2020.txt", 
                                   input_child_parent=False,
                                   output_child_parent=False) -> Tuple[LongTensor, LabelEncoder]:
    """
    Loads edges from a given tsv file into a PyTorch LongTensor.
    Meant for importing data where each edge appears as a line in the file, with
        <child_id>\t<parent_id>\t{} if input_child_parent is True
        <parent_id>\t<child_id>\t{} if input_child_parent is False
    :param edge_file: Path of dataset's hierarchy{_tc}.edge_list
    :returns: PyTorch LongTensor of edges with shape (num_edges, 2), LabelEncoder that numerized labels
    """
    start = time()
    logger.info(f"Loading edges from {edge_file}...")
    edges = pd.read_csv(edge_file, sep=" ", header=None).to_numpy()[:, :2]  # ignore line-final "{}"
    if input_child_parent != output_child_parent:
        edges[:, [0, 1]] = edges[:, [1, 0]]  # reverse parent-child to child-parent (or vice versa)
    le = LabelEncoder()
    edges = torch.tensor(le.fit_transform(edges.flatten()).reshape((-1,2)))
    logger.info(f"Loading complete, took {time() - start:0.1f} seconds")
    return edges, le


def name_id_mapping_from_file(name_id_file: Union[Path, str] = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/MeSH_name_id_mapping_2020.txt",
                              english=True) -> Dict:
    if english:
        SEP = "="
    else: # MESINESP
        SEP = "\t"
    with open(name_id_file, "r") as f:
        lines = f.readlines()

    name_id = dict()
    for i, line in enumerate(lines):
        if i == 0 and not english:
            continue
        try:
            name_id[line.split(SEP)[0].strip()] = line.split(SEP)[1].strip()
        except IndexError:
            breakpoint()

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


def collate_mlc_fn(batch):
    inputs, positives, negatives = batch

    positives = [[m for m in x["positives"]] for x in batch]
    max_pos_len = max(map(len, positives))
    positives = [p + [self.PAD] * (max_pos_len - len(p)) for p in positives]
    positives = torch.tensor(positives, dtype=torch.long)  # shape = (batch_size, num_positives)
    
    # xpositives = [[m for m in x["extra_positive_edges"]] for x in batch]
    # max_xpos_len = max(map(len, xpositives))
    # xpositives = [x + [self.PAD] * (max_xpos_len - len(x)) for x in xpositives]
    # xpositives = torch.tensor(xpositives, dtype=torch.long)  # shape = (batch_size, num_xpositives)

    negatives = [[m for m in x["negatives"]] for x in batch]
    max_neg_len = max(map(len, negatives))
    negatives = [n + [self.PAD] * (max_neg_len - len(n)) for n in negatives]
    negatives = torch.tensor(negatives, dtype=torch.long)  # shape = (batch_size, num_negatives)



@attr.s(auto_attribs=True)
class InstanceLabelsDataset(Dataset):
    """
    """

    instance_feats: Tensor
    labels: Tensor
    label_encoder: LabelEncoder  # label set accessable via label_encoder.classes_
    negative_sampler: Union[RandomNegativeEdges, HierarchicalNegativeEdges]

    def __attrs_post_init__(self):

        self._device = self.instance_feats.device
        self.instance_dim = self.instance_feats.shape[1]
        self.labels = self.prune_and_encode_labels_for_instances()
        
        # instance_label_pairs = []
        # for i, ls in enumerate(self.labels):
        #     instance_label_pairs.extend([i, l] for l in ls)
        # self.instance_label_pairs = torch.tensor(instance_label_pairs) 
        # self.instance_feats = torch.nn.Embedding.from_pretrained(self.instance_feats, freeze=True)

        self.negatives = []
        for ls in self.labels:
            self.negatives.append(self.get_negatives_for_labels(ls))
        breakpoint()

    def __getitem__(self, index: int) -> Dict:
        feats = self.instance_feats[index]
        labels = self.labels[index]
        negatives = self.negatives[index]
        ret = {
            "feats": feats,
            "positives": labels,
            "negatives": negatives,
        }
        return ret

    # def __getitem__(self, idxs: LongTensor) -> LongTensor:
    #     """
    #     :param idxs: LongTensor of shape (...,) indicating the index of the examples which to select
    #     :return: instance_feats of shape (batch_size, instance_dim), label_idxs of shape (batch_size, num_labels)
    #     """
    #     instance_idxs = self.instance_label_pairs[idxs][:, 0].to(self._device)
    #     label_idxs = self.instance_label_pairs[idxs][:, 1].to(self._device)
    #     instance_feats = self.instance_feats(instance_idxs)
    #     return instance_feats, label_idxs

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

    def get_negatives_for_labels(self, ls):
        negatives = set()
        for l in ls:
            negatives.update(sorted(list(set(self.negative_sampler.negative_roots[l].tolist()).difference({self.negative_sampler.EMB_PAD}))))
        return negatives

    @property
    def device(self):
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = device
        self.instance_feats = self.instance_feats.to(device)
        # self.labels = self.labels.to(device)
        return self


class CollateMeshFn(object):

    def __init__(self, tokenizer, train, num_labels):
        self.tokenizer = tokenizer
        self.train = train
        self.PAD = self.tokenizer.pad_token_id
        self.max_seq_len = min(1000, self.tokenizer.model_max_length)
        self.num_labels = num_labels
    
    def __call__(self, batch):

        # TODO add CLS token at beginning or end?
        tok_input = []
        for x in batch:
            if x["journal"]:
                journal = x["journal"]
            else:
                journal = ""
            if x["title"]:
                title = x["title"]
            else:
                title = ""
            if x["abstractText"]:
                abstractText = x["abstractText"]
            else:
                abstractText = ""
            tok_input.append(journal + f" {self.tokenizer.sep_token} " + title + f" {self.tokenizer.sep_token} " + abstractText)

        inputs = self.tokenizer(
            tok_input,
            return_tensors="pt",
            padding=True,  # pad_token_id = 1
            truncation=True,
            max_length=self.max_seq_len,
        )
        # logger.warning(f"max_length={self.max_seq_len}")
        # logger.warning(f"inputs['input_ids'].shape={inputs['input_ids'].shape}")

        if self.train:

            positives = [[m for m in x["positives"]] for x in batch]
            max_pos_len = max(map(len, positives))
            positives = [p + [self.PAD] * (max_pos_len - len(p)) for p in positives]
            positives = torch.tensor(positives, dtype=torch.long)  # shape = (batch_size, num_positives)
            
            xpositives = [[m for m in x["extra_positive_edges"]] for x in batch]
            max_xpos_len = max(map(len, xpositives))
            xpositives = [x + [self.PAD] * (max_xpos_len - len(x)) for x in xpositives]
            xpositives = torch.tensor(xpositives, dtype=torch.long)  # shape = (batch_size, num_xpositives)

            negatives = [[m for m in x["negatives"]] for x in batch]
            max_neg_len = max(map(len, negatives))
            negatives = [n + [self.PAD] * (max_neg_len - len(n)) for n in negatives]
            negatives = torch.tensor(negatives, dtype=torch.long)  # shape = (batch_size, num_negatives)

            return inputs, positives, negatives
        
        else:
            
            labels = [torch.tensor(list(set(x['positives']).union(x['extra_positive_edges']))) for x in batch]

            targets = []
            for i in range(len(labels)):
                targets.append(torch.zeros((self.num_labels,)).scatter_(0, labels[i], 1.0).tolist())
            targets = torch.tensor(targets)

            return inputs, targets


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

    file_path: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/allMeSH_2020.json"
    parent_child_mapping_path: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/MeSH_parent_child_mapping_2020.txt"
    name_id_mapping_path: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/MeSH_name_id_mapping_2020.txt"
    ancestors_cache_dir: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/cache/ancestors"
    negatives_cache_dir: str = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/cache/negatives"
    # TODO: DP: Wrap the dataset into a Shuffler instance to allow shuffling of the iterable dataset
    # https://pytorch.org/data/beta/generated/torchdata.datapipes.iter.Shuffler.html#torchdata.datapipes.iter.Shuffler

    train: bool = True
    huggingface_encoder: str = "microsoft/biogpt"
    negative_ratio: int = 500

    english: bool = True  # True means English BioASQ Task A, False means Spanish MESINESP

    def __attrs_post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_encoder)
        self.edges, self.le = edges_from_hierarchy_edge_list(
            edge_file=self.parent_child_mapping_path,
            input_child_parent=False,
            output_child_parent=False,
        )  # parent-child edge format for DiGraph
        self.name_id, self.id_name = name_id_mapping_from_file(
            name_id_file=self.name_id_mapping_path, english=self.english            
        )
        self.G = nx.DiGraph(
            self.edges.tolist()
        )
        logger.warning(f"num_nodes: {len(self.G.nodes())}")

        if self.english:
            self.ID_KEY = "pmid"
            self.LABELS_KEY = "meshMajor"
            self.ENCODING="windows-1252"
        else:  # MESINESP
            self.ID_KEY = "id"
            self.LABELS_KEY = "decsCodes"
            self.ENCODING="utf-8"

        self.collate_mesh_fn = CollateMeshFn(tokenizer=self.tokenizer, train=self.train, num_labels=len(self.G.nodes()))

    def label_to_id(self, labels: Iterable[str]) -> List[str]:
        anomalies = {'Respiratory Distress Syndrome, Adult': 'D012128'}  # 'Respiratory Distress Syndrome'
        # 'Pyruvate Dehydrogenase (Acetyl-Transferring) Kinase'
        ids = []
        for label in labels:
            try:
                ids.append(self.name_id[label])
            except KeyError:
                logger.warning(f'Label {label} not in self.name_id!')
                if label in anomalies:
                    ids.append(anomalies[label])
                else:
                    continue
        return ids

    def id_to_label(self, ids: Iterable[str]) -> List[str]:
        anomalies = {'anatomy_category': 'Anatomy',
                     'persons_category': 'Persons'}
        # Clostridium difficile
        labels = []
        for id in ids:
            try:
                labels.append(self.id_name[id])
            except KeyError:
                logger.warning(f'Id {id} not in self.id_name!')
                if id in anomalies:
                    labels.append(anomalies[id])
                else:
                    continue
        return labels

    def get_extra_positive_edges(
        self, labels: Iterable[str]
    ) -> Tuple[List[int], List[int]]:
        """Uses the MESH hierarchy to find extra positive edges for the labels.
        These are edges from positive children to positive ancestors.
        """
        if self.english:
            label_encs = self.le.transform(self.label_to_id(labels))
        else:
            label_encs = self.le.transform(labels)
        ancestor_encs = [nx.ancestors(self.G, l) for l in label_encs]
        ancestor_encs = set().union(*ancestor_encs)
        # ancestors = self.id_to_label(self.le.inverse_transform(list(ancestor_encs)))  # for debugging
        return ancestor_encs

    def read_set(self, pickled_set_fpath: str) -> Set:
        return pickle.load(open(pickled_set_fpath, "rb"))

    def get_negatives(self, positives: np.ndarray) -> np.ndarray:
            """Samples num_negatives labels from the mesh vocab that are not in labels"""
            # FIXME
            all_positives = set()
            all_negatives = set()
            for p in positives:
                # For each positive label, get the set of its ancestors
                anc = self.read_set(os.path.join(self.ancestors_cache_dir, f"{p}-ancestors.pkl"))
                anc.add(p)
                all_positives = all_positives.union(anc)
                negs = self.read_set(os.path.join(self.negatives_cache_dir, f"{p}-negatives.pkl"))
                all_negatives = all_negatives.union(negs)
            final_negatives = all_negatives.difference(all_positives)
            final_negatives = np.array(list(final_negatives))
            sampled_negatives = np.random.choice(final_negatives, size=(self.negative_ratio,))
            return sampled_negatives

    def prune_positives(self, positives: np.ndarray) -> np.ndarray:
        prune = set()
        for l1 in positives:
            if l1 not in prune:
                l1_ancestors = nx.ancestors(self.G, l1)
                for l2 in positives:
                    if l1 != l2:
                        # if l2 is an ancestor of l1, drop l2
                        if l2 in l1_ancestors:
                            prune.add(l2)
        keep = list(set(positives).difference(prune))
        return np.array(keep)

    def prune_positives_v2(self, positives: np.ndarray, ancestor_set: set) -> np.ndarray:
        return np.array(list(set(positives).difference(ancestor_set)))

    def parse_file(self, file_path, worker_id=0, num_workers=5):
        with open(file_path, encoding=self.ENCODING, mode="r") as f:
            next(f)  # skip   {"articles":[  line
            for i, line in enumerate(f):  # for article in ijson.items(f, "articles.item"):
                if (i - worker_id) % num_workers == 0:
                    # TODO parse line
                    line = line.strip().rstrip(",")
                    if line[-2:] == "]}" and self.english:
                        line = line[:-2]
                    article = json.loads(line)  # every intermediate line ends with ",", last line ends with "]}"
                    article = self.parse_article(article)
                    yield article
                else:
                    # TODO skip line
                    continue

    def parse_article(self, article):
        if self.english:
            article["positives"] = self.le.transform(self.label_to_id(article[self.LABELS_KEY]))
        else:
            article["positives"] = self.le.transform(article[self.LABELS_KEY])
        article["extra_positive_edges"] = self.get_extra_positive_edges(
            article[self.LABELS_KEY]
        )
        article["positives"] = self.prune_positives_v2(article["positives"], ancestor_set=article["extra_positive_edges"])
        if self.train:
            article["negatives"] = self.get_negatives(article["positives"])
        return article

    def get_stream(self, file_path):
        return self.parse_file(file_path)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_id = 0
            num_workers = 1
        else:  # in a worker process
            # split workload
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        return self.parse_file(file_path=self.file_path, worker_id=worker_id, num_workers=num_workers)


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
    bioasq = BioASQInstanceLabelsIterDataset(mesh_negative_sampler=MESHNegativeSampler())
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

    bioasq = BioASQInstanceLabelsIterDataset(mesh_negative_sampler=MESHNegativeSampler())
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


def check_no_two_labels_have_ancestor_relationship():
    bioasq = BioASQInstanceLabelsIterDataset()
    bioasq_iter = iter(bioasq)
    count_ancestors = 0
    count_no_ancestors = 0
    for i in range(10000):
        print(i)
        article = next(bioasq_iter)
        # print(f"processing article {article['pmid']}")
        ancestor_descendant_pair = False
        for label_a in article['positives']:
            for label_b in article['positives']:
                if label_b in nx.ancestors(bioasq.G, label_a):
                    ancestor_descendant_pair = True
                    descendant_label = bioasq.id_name[bioasq.le.inverse_transform([label_a]).item()]
                    ancestor_label = bioasq.id_name[bioasq.le.inverse_transform([label_b]).item()]
                    # print(f"article {article['pmid']} has descendant~ancestor pair {descendant_label} ~ {ancestor_label}")
        if ancestor_descendant_pair:
            count_ancestors += 1
        else:
            count_no_ancestors += 1
    breakpoint()


def convert_decs_obo_to_parent_child_format(decs_obo_fp="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/bioasq/MESINESP2/DeCS2020.obo",
                                            output_parent_child_fp="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/bioasq/MESINESP2/DeCS2020.parent_child_mapping.txt"):
    
    with open(decs_obo_fp, "r") as f:
        chunks = f.read().strip().split("\n\n")
        
    chunks = [c.strip().split("\n") for c in chunks[2:]]

    def process_chunk(c):
        id = [x for x in c if x.startswith('id: ')]
        assert len(id) == 1
        id = id[0][len('id: '):].strip('"')
        is_a = [x[len('is_a: '):].strip('"') for x in c if x.startswith('is_a: ')]
        return id, is_a

    id_to_is_a = list()
    for c in chunks:
        id, is_a = process_chunk(c)
        id_to_is_a.append((id, is_a))

    parent_child_mapping = []
    for child, parents in id_to_is_a:
        for parent in parents:
            parent_child_mapping.append((parent, child))

    with open(output_parent_child_fp, "w") as f:
        f.write("\n".join([" ".join(l) for l in parent_child_mapping]))


if __name__ == "__main__":
    # bioasq_test = BioASQInstanceLabelsIterDataset(mesh_negative_sampler=MESHNegativeSampler(), file_path="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/mesh/test.2020.json")
    # bioasq_test_iter = iter(bioasq_test)
    # print(next(bioasq_test_iter))

    # convert_decs_obo_to_parent_child_format()

    mesinesp = BioASQInstanceLabelsIterDataset(file_path="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/bioasq/MESINESP2/Subtrack1-Scientific_Literature/Train/training_set_subtrack1_all.json",
                                                  parent_child_mapping_path="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/bioasq/MESINESP2/DeCS2020.parent_child_mapping.txt",
                                                  name_id_mapping_path="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/bioasq/MESINESP2/DeCS2020.tsv",
                                                  english=False)
    mesinesp_iter = iter(mesinesp)
    print(next(mesinesp_iter))
