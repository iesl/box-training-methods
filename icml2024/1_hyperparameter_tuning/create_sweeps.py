import os
import copy
import argparse
import wandb

sweep_config_template = {
 "command":[
     "${env}",
     "${interpreter}",
     "${program}",
     "hyperparameter_tuning",
     "${args}" 
    ],
 "method": "bayes",
 "metric": {
  "goal": "maximize",
  "name": "[Eval] F1"
 },
 "name": "",            # this gets filled in by code below
 "parameters": {
    "learning_rate": {
         "distribution": "log_uniform",
         "max": 0,
         "min": -9.2
    },
    "negative_weight": {
        "distribution": "uniform",
        "max": 1,
        "min": 0
    },
 },
 "program": "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/scripts/box-training-methods"
}

graph_dir_paths = [
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=3-log_num_nodes=13-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=10-log_num_nodes=13-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=5-log_num_nodes=13-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=2-log_num_nodes=13-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True",
    "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True"
]


def parse_graph_dir_path(path):
    pieces = path.rstrip("/").split("/")
    graph_type, hparams = pieces[-2], pieces[-1]
    hparams_desc = " ".join(hparams.split("-"))
    hparams = [h.split("=") for h in hparams.split("-")]
    hparams = {h[0]: h[1] for h in hparams}
    ret = {
        "graph_type": graph_type,
        "graph_hyperparameters": hparams
    }
    desc = f"{graph_type} {hparams_desc}"
    return ret, desc


def main(args):

    # create 13 x 2 x 2 x 2 x 2 = 208 sweeps total
    sweep_ids = []
    for graph_dir_path in graph_dir_paths:  # 13
        d, s = parse_graph_dir_path(graph_dir_path)
        for model_type in ["vector_sim", "tbox"]:   # 2
            for negative_sampler in ["random", "hierarchical"]: # 2
                for negative_ratio in [4, 128]: # 2
                    for sample_positive_edges_from_tc_or_tr in ["tr", "tc"]:    # 2
                        sweep_config = copy.deepcopy(sweep_config_template)
                        sweep_config["name"] = f"{s} {model_type} {negative_sampler} negative_ratio={negative_ratio} sample_positive_edges_from_{sample_positive_edges_from_tc_or_tr}"
                        sweep_config["command"].append(f"--data_path={graph_dir_path}")
                        sweep_config["command"].append(f"--model_type={model_type}")
                        sweep_config["command"].append(f"--negative_sampler={negative_sampler}")
                        sweep_config["command"].append(f"--negative_ratio={negative_ratio}")
                        sweep_config["command"].append(f"--sample_positive_edges_from_tc_or_tr={sample_positive_edges_from_tc_or_tr}")
                        print(sweep_config)
                        id = wandb.sweep(sweep=sweep_config, entity="hierarchical-negative-sampling", project="icml2024")
                        print(f"Created wandb sweep with id {id}")
                        sweep_ids.append(id)

    with open(args.output_sweep_ids_file, "w") as f:
        f.write("\n".join(sweep_ids))


if __name__ == "__main__":
    
    # -------------------------------------------------------------
    # cd ./icml2024/1_vector_sim_hyperparameter_tuning/
    # python3 create_sweeps.py --output_sweep_ids_file ./sweep_ids.txt
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_sweep_ids_file", type=str, required=True,
                    default="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/1_vector_sim_hyperparameter_tuning/sweep_ids.txt",
                    help="txt file to store sweep ids of created sweeps")
    args = parser.parse_args()

    main(args)
