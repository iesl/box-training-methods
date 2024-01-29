import argparse
import wandb
import json
import numpy as np
import matplotlib.pyplot as plt


PATH = "hierarchical-negative-sampling/icml2024"


def plot(tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical, 
         title, fpath):

    tbox_random_means, tbox_random_stddevs = get_means_stddevs_stats(tbox_random)
    tbox_hierarchical_means, tbox_hierarchical_stddevs = get_means_stddevs_stats(tbox_hierarchical)
    vector_sim_random_means, vector_sim_random_stddevs = get_means_stddevs_stats(vector_sim_random)
    vector_sim_hierarchical_means, vector_sim_hierarchical_stddevs = get_means_stddevs_stats(vector_sim_hierarchical)

    plot_error_regions(
        means=[tbox_random_means, tbox_hierarchical_means, vector_sim_random_means, vector_sim_hierarchical_means],
        stds=[tbox_random_stddevs, tbox_hierarchical_stddevs, vector_sim_random_stddevs, vector_sim_hierarchical_stddevs],
        colors=['blue', 'red', 'blue', 'red'],
        labels=["T-Box:full", "T-Box:reduced", "Vector:full", "Vector:reduced"],
        markers=["s", "s", "+", "+"],
        title=title,
        fpath=fpath,
    )


def plot_error_regions(means, stds, colors, labels, markers, title, fpath):

    num_sets = len(means)

    if not all(len(means[i]) == len(stds[i]) for i in range(num_sets)):
        raise ValueError("Means and standard deviations must have the same length.")

    x_values = np.arange(len(means[0])) * 0.2

    plt.figure(figsize=(10, 6))

    for i in range(num_sets):
        mean = means[i]
        std = stds[i]
        color = colors[i]
        label = labels[i]
        marker = markers[i]

        plt.plot(x_values, mean, label=label, color=color, marker=marker)

        plt.fill_between(
            x_values,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2
        )

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(title)

    # plt.legend(title=f'Average reduction in + edges: {red_pos}%\nAverage reduction in - edges: {red_neg}%\nAverage reduction in total edges: {red_tot}%')
    # plt.legend()

    plt.grid(True)
    plt.savefig(fpath)
    plt.clf()


def get_means_stddevs_stats(runs):

    eval_f1s = []
    for r in runs:
        if r.state == 'finished':
            eval_f1s.append([0] + [f for f in list(r.history()['[Eval] F1']) if not np.isnan(f)])
        else:
            print("Non-finished run", r, f" in state {r.state}")
    eval_f1s = np.array(eval_f1s)
    mean = eval_f1s.mean(axis=0)
    std = eval_f1s.std(axis=0)
    
    # stats_json_fnames = [r.config['data_path'][:-len('.npz')] + '.icml2024stats.json' for r in runs]
    # stats_jsons = []
    # for fname in stats_json_fnames:
    #     with open(fname, 'r') as f:
    #         stats_jsons.append(json.load(f))

    # percentage_reductions_in_positive_edges = [1 - sj["[+edges_tr] / [+edges_tc]"] for sj in stats_jsons]
    # percentage_reductions_in_negative_edges = [1 - sj["[-edges_h] / [-edges_r]"] for sj in stats_jsons]
    # percentage_reductions_in_total_edges = [1 - sj["([+edges_tr] + [-edges_h]) / ([+edges_tc] + [-edges_r])"] for sj in stats_jsons]

    # avg_percentage_reduction_in_positive_edges = round(100 * (sum(percentage_reductions_in_positive_edges) / len(percentage_reductions_in_positive_edges)), 2)
    # avg_percentage_reduction_in_negative_edges = round(100 * (sum(percentage_reductions_in_negative_edges) / len(percentage_reductions_in_negative_edges)), 2)
    # avg_percentage_reduction_in_total_edges = round(100 * (sum(percentage_reductions_in_total_edges) / len(percentage_reductions_in_total_edges)), 2)
    
    return mean, std#, avg_percentage_reduction_in_positive_edges, avg_percentage_reduction_in_negative_edges, avg_percentage_reduction_in_total_edges


def get_runs_for_4_settings(sweep, tags, positive_with_random, positive_with_hierarchical):

    tbox_random = [r for r in sweep.runs if (tags | {"negative_sampler=random", "model_type=tbox", f"sample_positive_edges_from_tc_or_tr={positive_with_random}"}).issubset(set(r.tags))]
    vector_sim_random = [r for r in sweep.runs if (tags | {"negative_sampler=random", "model_type=vector_sim", f"sample_positive_edges_from_tc_or_tr={positive_with_random}"}).issubset(set(r.tags))]

    tbox_hierarchical = [r for r in sweep.runs if (tags | {"negative_sampler=hierarchical", "model_type=tbox", f"sample_positive_edges_from_tc_or_tr={positive_with_hierarchical}"}).issubset(set(r.tags))]
    vector_sim_hierarchical = [r for r in sweep.runs if (tags | {"negative_sampler=hierarchical", "model_type=vector_sim", f"sample_positive_edges_from_tc_or_tr={positive_with_hierarchical}"}).issubset(set(r.tags))]

    return tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical


def main():

    api = wandb.Api()

    for i, graph_type in enumerate(["balanced_tree", "nested_chinese_restaurant_process", "price"]):

        if i == 2:  # price sweep
            sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/z6t3y6oo")
        else:
            sweep = api.sweep(f"hierarchical-negative-sampling/icml2024/mpckfmmj")

        for negative_ratio in [4, 128]:
            for (positive_with_random, positive_with_hierarchical) in [("tr", "tr"), ("tr", "tc"), ("tc", "tr"), ("tc", "tc")]:

                tags = {f"graph_type={graph_type}", f"negative_ratio={negative_ratio}"}
                print(tags)
                print(positive_with_random, positive_with_hierarchical)
                
                fpath = f"./plots.v2/{graph_type}.k={negative_ratio}.+random={positive_with_random}.+hierarchical={positive_with_hierarchical}.png"

                if graph_type == "price": 
                    title = "Price's Network"
                elif graph_type == "nested_chinese_restaurant_process": 
                    title = "Nested Chinese Restaurant Process (nCRP)"
                elif graph_type == "balanced_tree": 
                    title = "Balanced Tree"

                tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical = get_runs_for_4_settings(sweep, tags, positive_with_random, positive_with_hierarchical)
                plot(tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical, title, fpath)


if __name__ == "__main__":
    
    main()
