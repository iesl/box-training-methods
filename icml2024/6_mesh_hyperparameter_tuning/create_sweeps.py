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
     "${args}",
     "--data_path=/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/mesh/MESH_2020.icml2024.FINAL.pt",
     "--mesh=1",
    ],
 "method": "random",
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
        "max": 1.0,
        "min": 0
    },
 },
 "program": "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/scripts/box-training-methods"
}


def main(args):

    # create 2 x 2 x 2 x 2 = 16 sweeps total
    sweep_ids = []
    # for model_type in ["vector_sim", "tbox"]:   # 2
    #     for negative_sampler in ["random", "hierarchical"]: # 2
    #         for negative_ratio in [4, 128]: # 2
    #             for sample_positive_edges_from_tc_or_tr in ["tr", "tc"]:    # 2
    for model_type in ["vector_sim"]:
        for negative_sampler in ["random"]:
            for negative_ratio in [128]:
                for sample_positive_edges_from_tc_or_tr in ["tc"]:
                    sweep_config = copy.deepcopy(sweep_config_template)
                    sweep_config["name"] = f"MeSH_2020.debug.v3 {model_type} {negative_sampler} negative_ratio={negative_ratio} sample_positive_edges_from_{sample_positive_edges_from_tc_or_tr}"
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
    # cd ./icml2024/6_mesh/
    # python3 create_sweeps.py --output_sweep_ids_file ./sweep_ids.txt
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_sweep_ids_file", type=str, required=True,
                    default="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/6_mesh/sweep_ids.txt",
                    help="txt file to store sweep ids of created sweeps")
    args = parser.parse_args()

    main(args)
