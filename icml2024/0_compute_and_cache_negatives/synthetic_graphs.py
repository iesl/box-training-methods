import os
import argparse
import time

from src.box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz, HierarchyAwareNegativeEdges


def compute_and_cache_negatives(npz_file):
    
    t1 = time.time()
    
    edges, num_nodes = edges_and_num_nodes_from_npz(npz_file)
    cache_dir = os.path.dirname(npz_file)
    selected_graph_name = os.path.basename(npz_file)[:-len(".npz")]
    H = HierarchyAwareNegativeEdges(edges=edges,
                                    negative_ratio=16,
                                    cache_dir=cache_dir,
                                    graph_name=selected_graph_name,
                                    is_tc=True)
    H.cache()
    
    print(f"Processed graph {selected_graph_name} in {str(time.time() - t1)} seconds")


def process_graph(args):
    graph_path = str(args.graph_path).strip('"')
    compute_and_cache_negatives(npz_file=graph_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--graph_path", type=str)
    args = parser.parse_args()

    process_graph(args)