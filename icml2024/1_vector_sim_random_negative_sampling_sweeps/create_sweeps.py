import os
import copy
import wandb

sweep_config_template = {
 "command":[
     "${env}",
     "${interpreter}",
     "${program}",
     "train_1",
     "${args}" 
    ],
 "method": "bayes",
 "metric": {
  "goal": "maximize",
  "name": "[Eval] F1"
 },
 "name": "",            # this gets filled in by code below
 "parameters": {
    "data_path": {
        "values": []    # this gets filled in by code below
    },
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
    "negative_ratio": {
        "values": [4, 128]
    },
    "seed": {   # mostly model seed (i.e. not graph seed)
        "values": [1, 2]       # FIXME what's the correct way to pick 10 random seed numbers?
    }
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

sweep_ids = []
for graph_dir_path in graph_dir_paths:
    
    # get all graph random seed npz files for this graph
    graph_npz_paths = []
    for file in os.listdir(graph_dir_path):
        if file.endswith(".npz"):
            graph_npz_paths.append(os.path.join(graph_dir_path, file))

    d, s = parse_graph_dir_path(graph_dir_path)
    sweep_config = copy.deepcopy(sweep_config_template)
    sweep_config["name"] = s
    sweep_config["parameters"]["data_path"]["values"] = graph_npz_paths
    print(sweep_config)
    id = wandb.sweep(sweep=sweep_config, entity="hierarchical-negative-sampling", project="icml2024")
    print(f"Created wandb sweep with id {id}")
    sweep_ids.append(id)

with open("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/1_vector_sim_random_negative_sampling_sweeps/sweep_ids.txt", "w") as f:
    f.write("\n".join(sweep_ids))
