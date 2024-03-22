import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


WANDB_PATH = "hierarchical-negative-sampling/icml2024"

DATA_PATHS = {
    "balanced_tree": [
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=3-log_num_nodes=13-transitive_closure=True/1439248948.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=10-log_num_nodes=13-transitive_closure=True/415728013.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=5-log_num_nodes=13-transitive_closure=True/1246911898.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=2-log_num_nodes=13-transitive_closure=True/1901635484.npz",
    ],
    "nested_chinese_restaurant_process": [
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/9.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/9.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/9.npz",
    ],
    "price": [
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/9.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/9.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/9.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/9.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/9.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/10.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/1.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/2.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/3.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/4.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/5.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/6.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/7.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/8.npz",
        "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/9.npz",
    ]
}


def hash_runs_in_sweep_by_data_path(sweep):
    print("hashing")
    hash_table = defaultdict(list)
    for r in sweep.runs:
        hash_table[r.config["data_path"]].append(r)
    print("done hashing")
    return hash_table


def graph_generative_hyperparameters_to_latex(graph_type, graph_generative_hyperparameters):
    graph_generative_hyperparameters.split("-")
    if graph_type == "balanced_tree":
        b = graph_generative_hyperparameters.split("-")[0].split("=")[1]
        return f'$b={b}$', f'b={b}'
    elif graph_type == "nested_chinese_restaurant_process":
        alpha = graph_generative_hyperparameters.split("-")[0].split("=")[1]
        return f'$\\alpha={alpha}$', f'alpha={alpha}'
    elif graph_type == "price":
        c = graph_generative_hyperparameters.split("-")[0].split("=")[1]
        gamma = graph_generative_hyperparameters.split("-")[1].split("=")[1]
        m = graph_generative_hyperparameters.split("-")[3].split("=")[1]
        return f'$c={c}, \gamma={gamma}, m={m}$', f"c={c}-gamma={gamma}-m={m}"


def plot_boxes_only():
    '''
    Boxes only: Pick some graphs (maybe one from each family, or maybe a few from balanced tree - eg. 3 balanced tree graphs, 1 ncrp, 2 price graphs)
    Show boxes on {E^tc, E^tr} x {\ol{E}, E_*}, with k=4
    '''
    
    api = wandb.Api()
    for i, graph_type in enumerate(["balanced_tree", "nested_chinese_restaurant_process", "price"]):
        
        # if i == 2:  # price sweep
        #     sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/z6t3y6oo")
        # else:
        #     sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/mpckfmmj")
        sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/4von71px")        

        data_path_to_runs = hash_runs_in_sweep_by_data_path(sweep)

        for data_path in DATA_PATHS[graph_type]:
            
            graph_generative_hyperparameters, graph_seed = data_path.rstrip(".npz").split("/")[-2:]
            hparams_latex, hparams_fname = graph_generative_hyperparameters_to_latex(graph_type, graph_generative_hyperparameters)

            plt.figure()
            
            runs = data_path_to_runs[data_path]

            four_xs, four_ys, four_labels, four_colors = [], [], [], ['b', 'g', 'r', 'c']
            for positive in ["tc", "tr"]:
                positive_label="$E^\mathrm{tc}$" if positive == "tc" else "$E^\mathrm{tr}$"

                for negative in ["random", "hierarchical"]:
                    negative_label="$\overline{E}$" if negative == "random" else "$E^-_{H^*}$"
                    
                    four_labels.append(f"({positive_label}, {negative_label})")

                    tags = {"model_type=tbox", f"graph_type={graph_type}", f"negative_ratio=4", f"sample_positive_edges_from_tc_or_tr={positive}", f"negative_sampler={negative}"}
                    run = [r for r in runs if r.config["data_path"] == data_path and tags.issubset(set(r.tags))]
                    assert len(run) == 1
                    run = run[0]
                    history = run.history()

                    if run.state == "finished":

                        total_examples = list(history['Total Examples'])
                        eval_f1s = list(history['[Eval] F1'])
                        assert len(eval_f1s) == len(total_examples)

                        total_examples_filtered = [0]
                        eval_f1s_filtered = [0]
                        for i, eval_f1 in enumerate(eval_f1s):
                            if not np.isnan(eval_f1):
                                if np.isnan(total_examples[i-1]):
                                    continue
                                eval_f1s_filtered.append(eval_f1)
                                total_examples_filtered.append(total_examples[i-1])

                        four_xs.append(total_examples_filtered)
                        four_ys.append(eval_f1s_filtered)

                    else:
                        continue

            try:
                assert len(four_xs) == 4
            except AssertionError:
                breakpoint()

            # truncate
            x_limit = min([max(xs) for xs in four_xs])
            plt.xlim(0, x_limit)

            for i in range(4):
                plt.plot(four_xs[i], four_ys[i], color=four_colors[i], label=four_labels[i])
            
            plt.grid(True)
            plt.legend()
            if graph_type == "balanced_tree":
                plt.title(f"Balanced Tree, {hparams_latex}")
            elif graph_type == "nested_chinese_restaurant_process":
                plt.title(f"nCRP, {hparams_latex}")
            elif graph_type == "price":
                plt.title(f"Price's Network, {hparams_latex}")
            plt.xlabel("Total Examples")
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ylabel("F1")
            plt.savefig(f"/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/4_plotting/plots.v3.DEBUGGED/boxes_only_plots/{graph_type}-{hparams_fname}.{graph_seed}.k=4.box_only.png")
            plt.clf()


def plot_boxes_vs_vectors():
    '''
    For point 1: Pick some representative graph (maybe ncrp?), k=128, show boxes with E^tr and E_*, and vectors with {E^tc, E^tr} x {\ol{E}, E_*}
    '''
    
    api = wandb.Api()
    for i, graph_type in enumerate(["balanced_tree", "nested_chinese_restaurant_process", "price"]):
        
        # if i == 2:  # price sweep
        #     sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/z6t3y6oo")
        # else:
        #     sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/mpckfmmj")
        sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/4von71px")

        data_path_to_runs = hash_runs_in_sweep_by_data_path(sweep)

        for data_path in DATA_PATHS[graph_type]:
            
            graph_generative_hyperparameters, graph_seed = data_path.rstrip(".npz").split("/")[-2:]
            hparams_latex, hparams_fname = graph_generative_hyperparameters_to_latex(graph_type, graph_generative_hyperparameters)

            plt.figure()
            
            runs = data_path_to_runs[data_path]

            # vector data
            #---------------------------------------------------------------------------------------------------
            four_xs, four_ys, four_labels, four_colors = [], [], [], ['b', 'g', 'r', 'c']
            for positive in ["tc", "tr"]:
                positive_label="$E^\mathrm{tc}$" if positive == "tc" else "$E^\mathrm{tr}$"

                for negative in ["random", "hierarchical"]:
                    negative_label="$\overline{E}$" if negative == "random" else "$E^-_{H^*}$"
                    
                    four_labels.append(f"({positive_label}, {negative_label})")

                    tags = {"model_type=vector_sim", f"graph_type={graph_type}", f"negative_ratio=128", f"sample_positive_edges_from_tc_or_tr={positive}", f"negative_sampler={negative}"}
                    run = [r for r in runs if r.config["data_path"] == data_path and tags.issubset(set(r.tags))]
                    assert len(run) == 1
                    run = run[0]
                    history = run.history()

                    if run.state == "finished":

                        total_examples = list(history['Total Examples'])
                        eval_f1s = list(history['[Eval] F1'])
                        assert len(eval_f1s) == len(total_examples)

                        total_examples_filtered = [0]
                        eval_f1s_filtered = [0]
                        for i, eval_f1 in enumerate(eval_f1s):
                            if not np.isnan(eval_f1):
                                if np.isnan(total_examples[i-1]):
                                    continue
                                eval_f1s_filtered.append(eval_f1)
                                total_examples_filtered.append(total_examples[i-1])

                        four_xs.append(total_examples_filtered)
                        four_ys.append(eval_f1s_filtered)

                    else:
                        continue
            #---------------------------------------------------------------------------------------------------
            
            
            # box data
            #---------------------------------------------------------------------------------------------------
            tags = {"model_type=tbox", f"graph_type={graph_type}", "negative_ratio=128", "sample_positive_edges_from_tc_or_tr=tr", "negative_sampler=hierarchical"}
            run = [r for r in runs if r.config["data_path"] == data_path and tags.issubset(set(r.tags))]

            assert len(run) == 1
            run = run[0]
            history = run.history()

            if run.state == "finished":

                total_examples = list(history['Total Examples'])
                eval_f1s = list(history['[Eval] F1'])
                assert len(eval_f1s) == len(total_examples)

                total_examples_filtered = [0]
                eval_f1s_filtered = [0]
                for i, eval_f1 in enumerate(eval_f1s):
                    if not np.isnan(eval_f1):
                        if np.isnan(total_examples[i-1]):
                            continue
                        eval_f1s_filtered.append(eval_f1)
                        total_examples_filtered.append(total_examples[i-1])

                box_xs = total_examples_filtered
                box_ys = eval_f1s_filtered

            else:
                continue
            #---------------------------------------------------------------------------------------------------

            # truncate
            x_limit = min([max(xs) for xs in four_xs + [box_xs]])
            plt.xlim(0, x_limit)

            for i in range(4):
                plt.plot(four_xs[i], four_ys[i], color=four_colors[i], label=four_labels[i], marker="+")
            plt.plot(box_xs, box_ys, color="k", label="($E^\mathrm{tr}$, $E^-_{H^*}$)", marker="s")
            
            plt.grid(True)
            plt.legend()
            if graph_type == "balanced_tree":
                plt.title(f"Balanced Tree, {hparams_latex}")
            elif graph_type == "nested_chinese_restaurant_process":
                plt.title(f"nCRP, {hparams_latex}")
            elif graph_type == "price":
                plt.title(f"Price's Network, {hparams_latex}")
            plt.xlabel("Total Examples")
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ylabel("F1")
            plt.savefig(f"/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/4_plotting/plots.v3.DEBUGGED/boxes_vs_vectors_plots/{graph_type}-{hparams_fname}.{graph_seed}.k=128.box_vs_vec.png")
            plt.clf()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["boxes_only", "boxes_vs_vectors"], type=str, required=True)
    args = parser.parse_args()

    if args.option == "boxes_only":
        plot_boxes_only()
    elif args.option == "boxes_vs_vectors":
        plot_boxes_vs_vectors()
