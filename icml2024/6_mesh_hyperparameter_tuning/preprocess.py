import argparse
import torch
import networkx as nx
import numpy as np
from scipy.sparse import save_npz, load_npz
from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz, HierarchyAwareNegativeEdges


PARENT_CHILD_MAPPING = "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/mesh/MeSH_parent_child_mapping_2020.txt"
NAME_ID_MAPPING = "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/mesh/MeSH_name_id_mapping_2020.txt"


def main():
    
    parent_child_lines = open(PARENT_CHILD_MAPPING, "r").readlines()
    parent_child_lines = [l.strip().split() for l in parent_child_lines]

    nodes = set()
    for u,v in parent_child_lines:
        nodes.add(u)
        nodes.add(v)
    nodes = sorted(list(nodes))

    num2node, node2num = dict(), dict()
    for i, node in enumerate(nodes):
        num2node[i] = node
        node2num[node] = i

    edges_nums = []
    nodes_nums = set()
    edges_mesh = []
    for u,v in parent_child_lines:
        edges_mesh.append([u, v])
        edges_nums.append([node2num[u], node2num[v]])
        nodes_nums.add(node2num[u])
        nodes_nums.add(node2num[v])
    nodes_nums = sorted(list(nodes_nums))

    # name_id_lines = open(NAME_ID_MAPPING, "r").readlines()
    # name_id_lines = [l.strip().split("=") for l in name_id_lines]

    graph_mesh = nx.DiGraph()
    graph_mesh.add_edges_from(edges_mesh)
    graph_nums = nx.DiGraph()
    graph_nums.add_edges_from(edges_nums)

    assert nx.is_directed_acyclic_graph(graph_mesh) # check DAG
    assert nx.is_directed_acyclic_graph(graph_nums)
    assert len([n for n,d in graph_mesh.in_degree() if d==0]) == 1  # check single root
    assert len([n for n,d in graph_nums.in_degree() if d==0]) == 1
    assert (torch.tensor(sorted(list(graph_nums.nodes))) == torch.tensor(nodes_nums)).all().item()  # check that nodes recovered from creating nx graph are the same
    assert (torch.tensor(nodes_nums) == torch.arange(len(nodes_nums))).all().item()     # check that node indices contiguous from 0 to num_nodes

    torch.save(torch.tensor(edges_nums), "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/mesh/MESH_2020.icml2024.FINAL.pt")

    breakpoint()

    H = HierarchyAwareNegativeEdges(
        edges=torch.tensor(edges_nums),
        negative_ratio=4,
        load_from_cache=False,
        cache_dir="/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/mesh",
        graph_name="MESH_2020.icml2024.FINAL"
    )
    H.cache()



if __name__ == "__main__":

    main()
