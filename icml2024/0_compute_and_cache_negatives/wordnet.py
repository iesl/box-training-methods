import os
import time

from src.box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz, HierarchyAwareNegativeEdges


if __name__ == "__main__":
    
    t1 = time.time()
    
    npz_file = "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/realworld/wordnet_full/wordnet_full.npz"
    edges, num_nodes = edges_and_num_nodes_from_npz(npz_file)
    cache_dir = os.path.dirname(npz_file)
    selected_graph_name = os.path.basename(npz_file)[:-len(".npz")]
    H = HierarchyAwareNegativeEdges(edges=edges,
                                    negative_ratio=16,
                                    cache_dir=cache_dir,
                                    graph_name=selected_graph_name,
                                    is_tc=False)
    H.cache()
    
    print(f"Processed graph {selected_graph_name} in {str(time.time() - t1)} seconds")
