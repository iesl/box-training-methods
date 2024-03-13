import argparse
from loguru import logger

import numpy as np

from box_training_methods.metrics import calculate_optimal_F1
from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz


GRAPHS_DIR = "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13"


def main(args):

    (graph_id, model_type, negative_ratio, negative_sampler, tc_or_tr, epoch) = args.prediction_scores_no_diag_npy.split("/")[-1].strip(".npy").split("|")
    graph_type, graph_hyperparameters, graph_seed = graph_id.split(".")
    graph_npz = f"{GRAPHS_DIR}/{graph_type}/{graph_hyperparameters}/{graph_seed}.npz"
    pos_index, num_nodes = edges_and_num_nodes_from_npz(graph_npz)
    ground_truth = np.zeros((num_nodes, num_nodes))
    ground_truth[pos_index[:, 0], pos_index[:, 1]] = 1
    ground_truth_no_diag = ground_truth[~np.eye(num_nodes, dtype=bool)]

    prediction_scores_no_diag = np.load(args.prediction_scores_no_diag_npy)

    metrics = calculate_optimal_F1(ground_truth_no_diag, prediction_scores_no_diag)
    breakpoint()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_scores_no_diag_npy", type=str, default="/scratch/workspace/wenlongzhao_umass_edu-hans/synthetic_graphs_prediction_scores_no_diag/balanced_tree.branching=10-log_num_nodes=13-transitive_closure=True.415728013|tbox|128|hierarchical|tc|10.npy")
    args = parser.parse_args()

    main(args)
