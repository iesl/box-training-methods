import os
import json
from collections import defaultdict



GRAPH_DIR_PATHS = ["/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=3-log_num_nodes=13-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=10-log_num_nodes=13-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=5-log_num_nodes=13-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/balanced_tree/branching=2-log_num_nodes=13-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=100-log_num_nodes=13-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=500-log_num_nodes=13-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/nested_chinese_restaurant_process/alpha=10-log_num_nodes=13-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=10-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=1-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.1-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/",
"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True/"]



def graph_hparams_to_str(graph_type, graph_hparams):
    if graph_type == "balanced_tree":
        return f"$b={graph_hparams['branching']}$"
    elif graph_type == "nested_chinese_restaurant_process":
        return f"$\\alpha={graph_hparams['alpha']}$"
    elif graph_type == "price":
        return f"$c={graph_hparams['c']}, \gamma={graph_hparams['gamma']}, m={graph_hparams['m']}$"


def main():

    header = ["Graph Type", "$\Theta$", "$\EE[|V|]$", "$\EE[|E|]$", "$\EE[|E^{\mathrm{tc}}|]$", "$\EE[|E^{\mathrm{tr}}|]$", "$\EE[|\overline E|]$", "$\EE[|E_{H^*}|]$", "$\EE[|E| / |\overline E|]$", "$\EE[|E^{\mathrm{tr}}| / |E^{\mathrm{tc}}|]$", "$\EE[|E_{H^*}| / |\overline E|]$", "$\EE[|\mathrm{reduced}| / |\mathrm{full}|]$"]

    graph_info = defaultdict(list)
    for graph_dir in GRAPH_DIR_PATHS:
        graph_type, graph_hparams = graph_dir.strip("/").split("/")[-2:]
        graph_hparams = {x.split("=")[0]: x.split("=")[1] for x in graph_hparams.split("-")}
        all_stats = [json.load(open(graph_dir + f, "r")) for f in os.listdir(graph_dir) if f.endswith(".icml2024stats.json")]
        if graph_type == "balanced_tree":
            assert len(all_stats) == 1
            stats = all_stats[0]
        else:
            assert len(all_stats) == 10
            keys = all_stats[0].keys()
            stats = dict()
            for k in keys:
                all_vals = [s[k] for s in all_stats]
                stats[k] = sum(all_vals) / len(all_vals)
        graph_info[graph_type].append((graph_hparams_to_str(graph_type, graph_hparams), stats))

    graph_info["wordnet"] = [("N/A", json.load(open("/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/realworld/wordnet_full/wordnet_full.icml2024stats.json", "r")))]

    table_dict = defaultdict(list)
    for graph_type in ["balanced_tree", "nested_chinese_restaurant_process", "price", "wordnet"]:
        hparams_stats = sorted(graph_info[graph_type], key=lambda x: x[0])
        for (hparams_str, stats) in hparams_stats:
            row = [hparams_str, stats["[nodes]"], stats["[+edges]"], stats["[+edges_tc]"], stats["[+edges_tr]"], stats["[-edges_r]"], stats["[-edges_h]"], str(round(100 * (stats["[+edges]"] / stats["[-edges_r]"]), 4)) + "\%", str(round(100 * stats["[+edges_tr] / [+edges_tc]"], 2)) + "\%", str(round(100 * stats["[-edges_h] / [-edges_r]"], 2)) + "\%", str(round(100 * stats["([+edges_tr] + [-edges_h]) / ([+edges_tc] + [-edges_r])"], 2)) + "\%"]
            row = [str(x) for x in row]
            table_dict[graph_type].append(row)

    prefix = "\\begin{landscape}\n\\begin{table}[p]\n\centering\n\\resizebox{1.2\\textwidth}{!}{%\n\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|}\n\hline"
    postfix = "\end{tabular}}\n\caption{Statistics for the synthetic transitively-closed DAGs plus WordNet used in our experiments (\Cref{sec:experiments}), counts for the number of positive/negative edges in the equivalent and optimal distinguishing sidigraphs, and the ratios for the $\mathrm{full}$ and $\mathrm{reduced}$ experimental settings. While graphs under Balanced Tree and WordNet are deterministic, the values of each entry for nCRP and Price are averaged over 10 random seeds, hence the expectation in the table header.}\n\label{tab:example}\n\end{table}\n\end{landscape}\n"
    header_str = ' & '.join(header) + '\\\\\n\hline'

    balanced_tree_str = '\multirow{4}{*}{Balanced Tree} '
    for row in table_dict['balanced_tree']:
        balanced_tree_str += ('& ' + ' & '.join(row) + '\\\\')
    balanced_tree_str += "\\hline"
    ncrp_str = '\multirow{4}{*}{nCRP} '
    for row in table_dict['nested_chinese_restaurant_process']:
        ncrp_str += ('& ' + ' & '.join(row) + '\\\\')
    ncrp_str += "\\hline"
    price_str = '\multirow{4}{*}{Price} '
    for row in table_dict['price']:
        price_str += ('& ' + ' & '.join(row) + '\\\\')
    price_str += "\\hline"
    wordnet_str = '\multirow{1}{*}{WordNet} '
    for row in table_dict['wordnet']:
        wordnet_str += ('& ' + ' & '.join(row) + '\\\\')
    wordnet_str += "\\hline"

    table_str = f"{prefix}\n{header_str}\n{balanced_tree_str}\n{ncrp_str}\n{price_str}\n{wordnet_str}\n{postfix}"

    print("\n\n")
    print(table_str)
    print()


if __name__ == "__main__":

    main()
