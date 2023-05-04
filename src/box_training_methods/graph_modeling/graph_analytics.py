import torch
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import time
import pickle

from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz#, HierarchicalNegativeEdges


def save_histogram(negatives_per_node, random_or_hierarchical, graph_id, save_dir):

    plt.hist(np.bincount(negatives_per_node), bins=np.arange(min(negatives_per_node).item(), max(negatives_per_node).item()))
    plt.title(f"{graph_id} {random_or_hierarchical}")
    plt.xlabel("# negatives")
    plt.ylabel("# nodes")
    filename = "/".join([save_dir, f"{graph_id}_{random_or_hierarchical}"]) + ".png"
    plt.savefig(filename)
    plt.clf()


def plot_node_histogram(negative_samples, save_path, num_ticks=10):
    # Create histogram
    fig, ax = plt.subplots()
    ax.hist(negative_samples, bins=range(min(negative_samples), max(negative_samples)+2), align='left', alpha=0.5)
    
    # Set x-axis ticks
    xticks = [int(min(negative_samples) + i * (max(negative_samples) - min(negative_samples)) / (num_ticks - 1)) for i in range(num_ticks)]
    ax.set_xticks(xticks)
    
    # Set axis labels and title
    ax.set_xlabel('Number of Negative Samples')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Negative Samples')
    
    # Show plot
    plt.savefig(save_path)
    plt.clf()


def plot_random_hns_diff_histogram(diffs, save_path, graph_id, num_ticks=10):

    # create x-axis values
    x_vals = range(len(diffs))
    
    # plot the line graph
    plt.plot(x_vals, diffs)
    
    # add labels and title
    plt.xlabel('Nodes in order of decreasing HNS gain')
    plt.ylabel('# fewer nodes required by HNS than random')
    plt.suptitle('HNS gains over Random Negative Sampling')
    plt.title(graph_id, fontsize=10)

    # save the plot
    plt.savefig(save_path)
    plt.clf()


def num_nodes_to_hns_roots(graph_npz_path, save_dir):

    graph_id = "-".join(graph_npz_path.split("/")[-3:])[:-len(".npz")]
    print(graph_id)

    negative_roots = torch.load(graph_npz_path.rstrip('.npz') + '.hns/negative_roots.pt')

    PAD = negative_roots.shape[0]
    node_hns_negative_samples = (negative_roots != PAD).int().sum(dim=-1)
    save_path = save_dir+graph_id+'.png'

    plot_node_histogram(node_hns_negative_samples, save_path)


def random_negatives_minus_hns_negatives(graph_npz_path, save_dir):

    graph_id = "-".join(graph_npz_path.split("/")[-3:])[:-len(".npz")]
    print(graph_id)

    negative_roots = torch.load(graph_npz_path.rstrip('.npz') + '.hns/negative_roots.pt')

    PAD = negative_roots.shape[0]
    node_hns_negative_samples = (negative_roots != PAD).int().sum(dim=-1)
    save_path = save_dir+graph_id+'.png'

    edges, num_nodes = edges_and_num_nodes_from_npz(graph_npz_path)
    G = nx.DiGraph()
    G.add_edges_from((edges).tolist())

    node_random_negative_samples = []
    for n in range(num_nodes):
        node_random_negative_samples.append(num_nodes - len([x for x in G.predecessors(n)]))
    node_random_negative_samples = torch.tensor(node_random_negative_samples)

    random_minus_hns_descending_order = sorted(list(node_random_negative_samples - node_hns_negative_samples), reverse=True)

    plot_random_hns_diff_histogram(random_minus_hns_descending_order, save_path, graph_id)


def graph_analytics(graph_npz_path, save_dir):

    graph_id = "-".join(graph_npz_path.split("/")[-3:])[:-len(".npz")]

    training_edges, num_nodes = edges_and_num_nodes_from_npz(graph_npz_path)

    HNE = HierarchicalNegativeEdges(
        edges=training_edges,
        negative_ratio=16,
        sampling_strategy="exact",
    )

    G = HNE.G
    density = nx.density(G)

    # RANDOM STATS
    num_rand_negatives_per_node = torch.tensor((num_nodes - HNE.A.sum(axis=0))).squeeze()       # everybody but parents is a possible random negative parent
    max_num_rand_negatives = torch.max(num_rand_negatives_per_node).item()
    min_num_rand_negatives = torch.min(num_rand_negatives_per_node).item()
    avg_num_rand_negatives = torch.mean(num_rand_negatives_per_node.float()).item()
    # save_histogram(negatives_per_node=num_rand_negatives_per_node,
    #                random_or_hierarchical="random",
    #                graph_id=graph_id,
    #                save_dir=save_dir)

    # HIERARCHICAL STATS
    num_hier_negative_roots_per_node = (HNE.negative_roots != HNE.EMB_PAD).int().sum(dim=-1)  # TODO save in file as histogram (wandb or run dir)
    max_num_hier_negative_roots = HNE.negative_roots.shape[-1]
    min_num_hier_negative_roots = torch.min(num_hier_negative_roots_per_node).item()
    avg_num_hier_negative_roots = torch.mean(num_hier_negative_roots_per_node.float()).item()
    # save_histogram(negatives_per_node=num_hier_negative_roots_per_node,
    #                random_or_hierarchical="hierarchical",
    #                graph_id=graph_id,
    #                save_dir=save_dir)

    # the greater this is the more efficient hierarchical sampling will be
    avg_rand_to_avg_hier_ratio = avg_num_rand_negatives / avg_num_hier_negative_roots

    roots = [n for n, d in G.in_degree() if d == 0]

    t1 = time.time()
    max_depths = []
    for r in roots:
        max_depths.append(dfs_max_depth(r, G, 0))
    max_depth = max(max_depths)
    t2 = time.time()
    print(f"Time dfs: {str(t2 - t1)}")

    t3 = time.time()
    depths = []
    for r in roots:
        for n in G.nodes:
            try:
                depths.append(len(nx.shortest_path(G, source=r, target=n)) - 1)
            except nx.NetworkXNoPath:
                pass
    max_depth = max(depths)
    t4 = time.time()
    print(f"Time brute force: {str(t4 - t3)}")

    breakpoint()

    stats = {
        "graph_id": graph_id,
        "graph_density": density,
        "# nodes": num_nodes,
        "max depth": max_depth,
        "random max # negatives": max_num_rand_negatives,
        "random min # negatives": min_num_rand_negatives,
        "random avg # negatives": avg_num_rand_negatives,
        "hierarchical max # negative roots": max_num_hier_negative_roots,
        "hierarchical min # negative roots": min_num_hier_negative_roots,
        "hierarchical avg # negative roots": avg_num_hier_negative_roots,
        "avg random to avg hierarchical ratio": avg_rand_to_avg_hier_ratio,
    }

    with open("/".join([save_dir, graph_id]) + ".json", "w") as f:
        json.dump(stats, f, sort_keys=False, indent=4)

    return stats


def dfs_max_depth(r, G, max_depth):
    children_max_depths = []
    print(f"r: {r}")
    print(f"\tmax depth: {max_depth}")
    for s in G.successors(r):
        children_max_depths.append(dfs_max_depth(s, G, max_depth + 1))
    print(f"\tchildren max depths: {children_max_depths}")
    return max(children_max_depths) if len(children_max_depths) > 0 else max_depth


def all_stats_to_csv(all_stats, csv_fpath):

    rows = []
    header_row = ",".join(all_stats[0].keys())
    rows.append(header_row)
    for stats in all_stats:
        values = stats.values()
        row = ",".join(values)
        rows.append(row)

    csv_str = "\n".join(rows)
    with open(csv_fpath, "w") as f:
        f.write(csv_str)


def generate_analytics_for_graphs_in_dir(graphs_root="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13/",
                                         save_dir="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graph_analytics/hns_histograms/"):

    all_stats = []
    for root, dirs, files in os.walk(graphs_root):
        print(dirs)
        # for f in files:
        #     print(f)
            # if f.endswith(".npz"):

            #     if 'kronecker_graph' in root or 'scale_free_network' in root:
            #         continue

            #     graph_npz_path = "/".join([root, f])
            #     # num_nodes_to_hns_roots(graph_npz_path, save_dir)
            #     random_negatives_minus_hns_negatives(graph_npz_path, save_dir)
            #     # stats = graph_analytics(graph_npz_path, save_dir)
            #     # all_stats.append(stats)

    return all_stats


if __name__ == '__main__':

    ### NON-TC GRAPHS
    #   'data/graphs13/balanced_tree/branching=10-log_num_nodes=13-transitive_closure=False', 
    #   'data/graphs13/balanced_tree/branching=5-log_num_nodes=13-transitive_closure=False', 
    #   'data/graphs13/balanced_tree/branching=2-log_num_nodes=13-transitive_closure=False',
    #   'data/graphs13/balanced_tree/branching=3-log_num_nodes=13-transitive_closure=False',
    #   'data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=False',
    #   'data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=False',
    #   'data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=False',
    #   'data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=False',
    #   'data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=False',
    #   'data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=False',
    #   'data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=False',
    #   'data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=False',
    #   'data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=False'


    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/kronecker_graph/a=1.0-b=0.6-c=0.5-d=0.2-log_num_nodes=12-transitive_closure=False/1619702443.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/hierarchical_negative_sampling_debugging_graphs/log_num_nodes=12-transitive_closure=False-which=dag/1160028402.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/balanced_tree/branching=2-log_num_nodes=12-transitive_closure=False/2952040816.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    # graph_analytics("/Users/brozonoyer/Desktop/IESL/box-training-methods/data/graphs/hierarchical_negative_sampling_debugging_graphs/log_num_nodes=12-transitive_closure=False-which=balanced-tree/1196640715.npz",
    #                 save_dir="/Users/brozonoyer/Desktop/IESL/box-training-methods/figs/graph_analytics/")
    all_stats = generate_analytics_for_graphs_in_dir(graphs_root="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13/",
                                                     save_dir="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graph_analytics/hns_histograms.v2/")
    # all_stats_to_csv(all_stats, "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graph_analytics/graphs13_stats.csv")
