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
                                    aggressive_pruning=True,
                                    negative_ratio=16,
                                    cache_dir=cache_dir,
                                    selected_graph_name=selected_graph_name)
    H.cache()
    
    print(f"Processed graph {selected_graph_name} in {str(time.time() - t1)} seconds")


def process_graph_dir(args):
    graph_dir = str(args.graph_dir).strip('"')  # FIXME need to strip " because bash script adds it
    for file in os.listdir(graph_dir):
        if file.endswith(".npz"):
            compute_and_cache_negatives(npz_file=os.path.join(graph_dir, file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--graph_dir", type=str)
    args = parser.parse_args()

    process_graph_dir(args)
