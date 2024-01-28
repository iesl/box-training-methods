import argparse
import json
import numpy as np

from loguru import logger

from box_training_methods.metrics import calculate_optimal_F1
from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz


def main(args):
    
    logger.debug("a")
    prediction_scores_no_diag = np.load(args.prediction_scores_no_diag_npy)
    logger.debug("b")
    ground_truth = np.zeros((82115, 82115))
    logger.debug("c")
    pos_index, _ = edges_and_num_nodes_from_npz("/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/realworld/wordnet_full/wordnet_full.npz")
    logger.debug("d")
    ground_truth[pos_index[:, 0], pos_index[:, 1]] = 1
    logger.debug("e")
    ground_truth_no_diag = ground_truth[~np.eye(82115, dtype=bool)]
    logger.debug("f")
    metrics = calculate_optimal_F1(ground_truth_no_diag, prediction_scores_no_diag)
    logger.debug("g")
    ckpt_info = args.prediction_scores_no_diag_npy.split("/")[-1].strip(".npy")
    logger.debug("h")
    with open(f"/scratch/workspace/wenlongzhao_umass_edu-hans/icml2024_wordnet_f1/{ckpt_info}.f1.json", "w") as f:
        json.dump(metrics, f)
    logger.debug("i")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_scores_no_diag_npy", type=str)
    args = parser.parse_args()

    main(args)
