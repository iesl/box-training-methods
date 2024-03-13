import os
import argparse
import time

from src.box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz, HierarchyAwareNegativeEdges


def check_compatibility(npz_file):
    
    t1 = time.time()
    
    edges, num_nodes = edges_and_num_nodes_from_npz(npz_file)
    cache_dir = os.path.dirname(npz_file)
    selected_graph_name = os.path.basename(npz_file)[:-len(".npz")]
    
    H1 = HierarchyAwareNegativeEdges(edges=edges,
                                     negative_ratio=16,
                                     cache_dir=cache_dir,
                                     graph_name=selected_graph_name,
                                     is_tc=True,
                                     load_from_cache=True)
    
    H2 = HierarchyAwareNegativeEdges(edges=edges,
                                     negative_ratio=16,
                                     cache_dir=cache_dir,
                                     graph_name=selected_graph_name,
                                     is_tc=True,
                                     load_from_cache=False)

    compatible = (H1.negative_edges == H2.negative_edges).all().item()
    print(compatible)
    with open(os.path.join(cache_dir, selected_graph_name + ".negativescompatible"), "w") as f:
        f.write(str(compatible))


def process_graph(args):
    graph_path = str(args.graph_path).strip('"')
    check_compatibility(npz_file=graph_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--graph_path", type=str)
    args = parser.parse_args()

    process_graph(args)
