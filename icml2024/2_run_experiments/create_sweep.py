import wandb
import argparse


def main(args):

    sweep_config = {
    "command":[
        "${env}",
        "${interpreter}",
        "${program}",
        f"train_{args.model}",      # train_tbox or train_vector_sim
        "${args}",
    ],
    "method": "grid",
    "metric": {
    "goal": "maximize",
    "name": "[Eval] F1"
    },
    "name": args.model,             # tbox or vector_sim
    "parameters": {
        "data_path": {
            "values": [
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/balanced_tree/branching=3-log_num_nodes=13-transitive_closure=True/1439248948.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/balanced_tree/branching=10-log_num_nodes=13-transitive_closure=True/415728013.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/balanced_tree/branching=5-log_num_nodes=13-transitive_closure=True/1246911898.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/balanced_tree/branching=2-log_num_nodes=13-transitive_closure=True/1901635484.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/9.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/9.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/9.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/9.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/9.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/9.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/9.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/9.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/10.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/1.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/2.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/3.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/4.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/5.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/6.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/7.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/8.npz",
                "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-datagraphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/9.npz",     
            ]
        },
        "negative_ratio": {
            "values": [4, 128]
        },
        "negative_sampler": {
            "values": ["hierarchical", "random"]
        },
        "sample_positive_edges_from_tc_or_tr": {
            "values": ["tc", "tr"]
        },
    },
    "program": "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/scripts/box-training-methods"
    }

    if args.model == "vector_sim":
        sweep_config["command"].append(f"--lr_nw_json={args.lr_nw_json}")

    sweep_id = wandb.sweep(sweep=sweep_config, entity="hierarchical-negative-sampling", project="icml2024")
    print(sweep_id)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["tbox", "vector_sim"], required=True,
                        help="whether to run tbox or vector_sim sweep")
    parser.add_argument("--lr_nw_json", type=str, help="json file with graph type and negative ratio to best learning rate and negative weight",
                        default="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/1_vector_sim_hyperparameter_tuning/graph_type_to_best_learning_rate_and_negative_weight.json")
    args = parser.parse_args()

    main(args)
