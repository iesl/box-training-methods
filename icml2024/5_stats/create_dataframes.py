import os, json
import wandb
import pandas as pd
import argparse


def main1(graph_type):

    api = wandb.Api()

    columns = ["graph_unique_id", "model_type", "full_positive_edge_set", "full_negative_edge_set", "num_negative_samples", "step", "total_examples", "F1", "AUC", "threshold"]
    rows = []

    # if graph_type == "price":
    #     sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/z6t3y6oo")
    # else:
    #     sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/mpckfmmj")
    sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/4von71px")

    for model_type in ["tbox", "vector_sim"]:
        for positive_edge_set in ["tr", "tc"]:
            full_positive_edge_set = positive_edge_set == "tc"                    
            for negative_edge_set in ["hierarchical", "random"]:
                full_negative_edge_set = negative_edge_set == "random"
                for negative_ratio in [4, 128]:

                        tags = {f"graph_type={graph_type}", f"negative_ratio={negative_ratio}", f"model_type={model_type}", f"sample_positive_edges_from_tc_or_tr={positive_edge_set}", f"negative_sampler={negative_edge_set}"}
                        print(tags)
                        runs = [r for r in sweep.runs if tags.issubset(set(r.tags))]
                        for r in runs:
                            if r.state == 'finished':
                                graph_unique_id = "/".join(r.config['data_path'].rstrip(".npz").split("/")[-3:])
                                history = r.history()
                                for step, info in history.iterrows():
                                    total_examples = info["Total Examples"]
                                    f1 = info["[Eval] F1"]
                                    auc = info["[Eval] AUC"]
                                    threshold = info["[Eval] threshold"]
                                    row = [graph_unique_id, model_type, full_positive_edge_set, full_negative_edge_set, negative_ratio, step, total_examples, f1, auc, threshold]
                                    rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(f"/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/5_stats/runs_stats.DEBUGGED/runs_stats.{graph_type}.csv")


def main2():

    rows = []
    for subdir, dirs, files in os.walk("/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/"):
        for file in files:
            file_to_check = os.path.join(subdir, file)
            if file_to_check.endswith(".icml2024stats.json"):
                # print(subdir)
                graph_family = subdir.split("/")[-2]
                graph_generative_hyperparameters = subdir.split("/")[-1]
                graph_seed = file[:-len(".icml2024stats.json")]
                graph_unique_id = "/".join([graph_generative_hyperparameters, graph_seed])
                if graph_unique_id == "c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/1":
                    print("DEBUG", file_to_check)
                # print(graph_family, graph_generative_hyperparameters, graph_seed)
                with open(file_to_check, "r") as f:
                    stats = json.load(f)
                row = [graph_unique_id, graph_family, graph_generative_hyperparameters, graph_seed, stats["[nodes]"], stats["[+edges_tc]"], stats["[+edges_tr]"], stats["[-edges_r]"], stats["[-edges_h]"]]
                rows.append(row)

    columns = ["graph_unique_id", "graph_family", "graph_generative_hyperparameters", "graph_seed", "num_nodes", "size_E=E^tc", "size_E^tr", "size_overline_E", "size_E^*"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/5_stats/graph_stats.csv")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["1", "2"], required=True)
    parser.add_argument("--graph_type", default="balanced_tree", choices=["balanced_tree", "nested_chinese_restaurant_process", "price"], required=False)
    args = parser.parse_args()

    if args.which == "1":
        main1(args.graph_type)
    else:
        main2()
