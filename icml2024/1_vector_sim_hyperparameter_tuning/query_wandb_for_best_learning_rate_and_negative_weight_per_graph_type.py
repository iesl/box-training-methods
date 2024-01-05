import wandb
import argparse
import json
from collections import defaultdict


def main(args):

    api = wandb.Api()

    sweep_ids = [l.strip() for l in open(args.input_sweep_ids_file, "r").readlines()]
    
    output_json = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    for sweep_id in sweep_ids:
        
        sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/{sweep_id}")
        runs = sweep.runs
        
        data_path = [x for x in sweep.config['command'] if x.startswith('--data_path=')][0]
        sweep_graph_type, sweep_graph_hparams = data_path.rstrip('/').split('/')[-2:]
        sweep_negative_ratio = [x for x in sweep.config['command'] if x.startswith('--negative_ratio=')][0][len('--negative_ratio='):]

        # best_run = max(runs, key=lambda r: r.summary['[Eval] F1'])
        best_run = sweep.best_run()
        best_run_config = json.loads(best_run.json_config)
        best_run_learning_rate = best_run_config['learning_rate']['value']
        best_run_negative_weight = best_run_config['negative_weight']['value']
        
        output_json[sweep_graph_type][sweep_graph_hparams][f'negative_ratio={sweep_negative_ratio}']['best_learning_rate'] = best_run_learning_rate
        output_json[sweep_graph_type][sweep_graph_hparams][f'negative_ratio={sweep_negative_ratio}']['best_negative_weight'] = best_run_negative_weight

    with open(args.output_json_file, "w") as f:
        json.dump(output_json, f, sort_keys=True, indent=4)


if __name__ == "__main__":

    # ----------—----------—----------—----------—
    # python3 query_wandb_for_best_learning_rate_and_negative_weight_per_graph_type.py --input_sweep_ids_file ./sweep_ids.txt --output_json_file ./graph_type_to_best_learning_rate_and_negative_weight.json
    # ----------—----------—----------—----------—

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_sweep_ids_file", type=str, required=True,
                        default="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/1_vector_sim_hyperparameter_tuning/sweep_ids.txt",
                        help="txt file with sweep ids")
    parser.add_argument("--output_json_file", type=str, required=True,
                        default="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/1_vector_sim_hyperparameter_tuning/best_learning_rate_and_negative_weight_per_graph_type.json",
                        help="json file where to write graph type and negative ratio to best learning rate and negative weight")
    args = parser.parse_args()

    main(args)
