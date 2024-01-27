import argparse
import json
import numpy as np

from box_training_methods.metrics import calculate_optimal_F1
from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz


def main(args):
    
    prediction_scores_no_diag = np.load(args.prediction_scores_no_diag_npz)['prediction_scores_no_diag']
    ground_truth = np.zeros((82115, 82115))
    pos_index, _ = edges_and_num_nodes_from_npz("/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/realworld/wordnet_full/wordnet_full.npz")
    ground_truth[pos_index[:, 0], pos_index[:, 1]] = 1
    ground_truth_no_diag = ground_truth[~np.eye(82115, dtype=bool)]

    metrics = calculate_optimal_F1(ground_truth_no_diag, prediction_scores_no_diag)

    ckpt_info = args.prediction_scores_no_diag_npz.split("/")[-1].strip(".npz")
    with open(f"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/icml2024_wordnet_f1/{ckpt_info}.f1.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_scores_no_diag_npz", type=str)
    args = parser.parse_args

    main(args)
