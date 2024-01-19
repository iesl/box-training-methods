import argparse
import wandb
import json
import numpy as np
import matplotlib.pyplot as plt


PATH = "hierarchical-negative-sampling/icml2024"


def plot(tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical, title, fpath):
    tbox_random_means, tbox_random_stddevs, avg_percentage_reduction_in_negative_edges = get_means_stddevs_stats(tbox_random, get_avg_percentage_reduction_in_negative_edges=True)
    tbox_hierarchical_means, tbox_hierarchical_stddevs = get_means_stddevs_stats(tbox_hierarchical)
    vector_sim_random_means, vector_sim_random_stddevs = get_means_stddevs_stats(vector_sim_random)
    vector_sim_hierarchical_means, vector_sim_hierarchical_stddevs = get_means_stddevs_stats(vector_sim_hierarchical)
    plot_error_regions(
        means=[tbox_random_means, tbox_hierarchical_means, vector_sim_random_means, vector_sim_hierarchical_means],
        stds=[tbox_random_stddevs, tbox_hierarchical_stddevs, vector_sim_random_stddevs, vector_sim_hierarchical_stddevs],
        colors=['blue', 'orange', 'green', 'red'],
        labels=["T-Box:random", "T-Box:hierarchical", "Vector:random", "Vector:hierarchical"],
        title=title,
        fpath=fpath,
        avg_percentage_reduction_in_neg_edges=avg_percentage_reduction_in_negative_edges,
    )


def plot_error_regions(means, stds, colors, labels, title, fpath, avg_percentage_reduction_in_neg_edges):
    """
    Plot means with shaded error regions.

    Parameters:
    - means: List of arrays containing mean values for each set.
    - stds: List of arrays containing standard deviation values for each set.
    - colors: List of colors for each set.
    - labels: List of labels for each set.

    Returns:
    - None
    """
    num_sets = len(means)

    if not all(len(means[i]) == len(stds[i]) for i in range(num_sets)):
        raise ValueError("Means and standard deviations must have the same length.")

    x_values = np.arange(len(means[0]))

    plt.figure(figsize=(10, 6))

    for i in range(num_sets):
        mean = means[i]
        std = stds[i]
        color = colors[i]
        label = labels[i]

        plt.plot(x_values, mean, label=label, color=color)

        plt.fill_between(
            x_values,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2
        )

    plt.xlabel("Step")
    plt.ylabel("[Eval] F1")
    plt.title(title)
    if "balanced_tree" in title:
        plt.legend(title=f'Reduction in negative edges: {avg_percentage_reduction_in_neg_edges}%')
    else:
        plt.legend(title=f'Average reduction in negative edges: {avg_percentage_reduction_in_neg_edges}%')
    plt.grid(True)
    plt.savefig(fpath)
    plt.clf()


def get_means_stddevs_stats(runs, get_avg_percentage_reduction_in_negative_edges=False):
    eval_f1s = []
    for r in runs:
        if r.state == 'finished':
            eval_f1s.append([f for f in list(r.history()['[Eval] F1']) if not np.isnan(f)])
        else:
            print("Non-finished run", r, f" in state {r.state}")
    eval_f1s = np.array(eval_f1s)
    mean = eval_f1s.mean(axis=0)
    std = eval_f1s.std(axis=0)
    
    if get_avg_percentage_reduction_in_negative_edges:
        stats_json_fnames = [r.config['data_path'][:-len('.npz')] + '.icml2024stats.json' for r in runs]
        stats_jsons = []
        for fname in stats_json_fnames:
            with open(fname, 'r') as f:
                stats_jsons.append(json.load(f))
        percentage_reductions_in_negative_edges = [1 - sj["[-edges_h] / [-edges_r]"] for sj in stats_jsons]
        avg_percentage_reduction_in_negative_edges = round(100 * (sum(percentage_reductions_in_negative_edges) / len(percentage_reductions_in_negative_edges)), 2)
        return mean, std, avg_percentage_reduction_in_negative_edges
    
    return mean, std


def main(args):

    api = wandb.Api()
    sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/{args.sweep_id}")

    for negative_ratio in [4, 128]:
        for sample_positive_edges_from_tc_or_tr in ["tc", "tr"]:
            base_tags = {f"negative_ratio={negative_ratio}", f"sample_positive_edges_from_tc_or_tr={sample_positive_edges_from_tc_or_tr}"}
            base_fpath = f"./plots/negative_ratio={negative_ratio}_{sample_positive_edges_from_tc_or_tr}"
            for graph_type in ["price", "nested_chinese_restaurant_process", "balanced_tree"]:

                if graph_type == "price":
                    for c in [0.01, 0.1]:
                        for gamma in [1.0]:
                            for m in [1, 5, 10]:
                                tags = base_tags | {"graph_type=price", f"c={c}", f"gamma={gamma}", f"m={m}"}
                                print(tags)
                                title = rf"Price's network $c={c}$ $\gamma={gamma}$ $m={m}$"
                                fpath = f"{base_fpath}/price_c={c}_gamma={gamma}_m={m}.png"
                                tbox_random = [r for r in sweep.runs if (tags | {"negative_sampler=random", "model_type=tbox"}).issubset(set(r.tags))]
                                tbox_hierarchical = [r for r in sweep.runs if (tags | {"negative_sampler=hierarchical", "model_type=tbox"}).issubset(set(r.tags))]
                                vector_sim_random = [r for r in sweep.runs if (tags | {"negative_sampler=random", "model_type=vector_sim"}).issubset(set(r.tags))]
                                vector_sim_hierarchical = [r for r in sweep.runs if (tags | {"negative_sampler=hierarchical", "model_type=vector_sim"}).issubset(set(r.tags))]
                                plot(tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical, title, fpath)

                if graph_type == "nested_chinese_restaurant_process":
                    for alpha in [10, 100, 500]:
                        tags = base_tags | {"graph_type=nested_chinese_restaurant_process", f"alpha={alpha}"}
                        print(tags)
                        title = rf"nCRP $\alpha={alpha}$"
                        fpath = f"{base_fpath}/ncrp_alpha={alpha}.png"
                        tbox_random = [r for r in sweep.runs if (tags | {"negative_sampler=random", "model_type=tbox"}).issubset(set(r.tags))]
                        tbox_hierarchical = [r for r in sweep.runs if (tags | {"negative_sampler=hierarchical", "model_type=tbox"}).issubset(set(r.tags))]
                        vector_sim_random = [r for r in sweep.runs if (tags | {"negative_sampler=random", "model_type=vector_sim"}).issubset(set(r.tags))]
                        vector_sim_hierarchical = [r for r in sweep.runs if (tags | {"negative_sampler=hierarchical", "model_type=vector_sim"}).issubset(set(r.tags))]
                        plot(tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical, title, fpath)

                if graph_type == "balanced_tree":
                    for branching in [2, 3, 5, 10]:
                        tags = base_tags | {"graph_type=balanced_tree", f"branching={branching}"}
                        print(tags)
                        title = rf"Balanced Tree $b={branching}$"
                        fpath=f"{base_fpath}/balanced-tree_branching={branching}.png"
                        tbox_random = [r for r in sweep.runs if (tags | {"negative_sampler=random", "model_type=tbox"}).issubset(set(r.tags))]
                        tbox_hierarchical = [r for r in sweep.runs if (tags | {"negative_sampler=hierarchical", "model_type=tbox"}).issubset(set(r.tags))]
                        vector_sim_random = [r for r in sweep.runs if (tags | {"negative_sampler=random", "model_type=vector_sim"}).issubset(set(r.tags))]
                        vector_sim_hierarchical = [r for r in sweep.runs if (tags | {"negative_sampler=hierarchical", "model_type=vector_sim"}).issubset(set(r.tags))]
                        plot(tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical, title, fpath)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, help="synthetic graphs sweep id", default="mpckfmmj", required=True)
    args = parser.parse_args()

    main(args)
