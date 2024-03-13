import os, json
from collections import defaultdict
from matplotlib import pyplot as plt


F1_PATH = "/scratch/workspace/wenlongzhao_umass_edu-hans/icml2024_wordnet_f1/"


def collect_f1s():
    f1s = defaultdict(lambda: defaultdict(int))
    for f1_file in os.listdir(F1_PATH):
        info = f1_file.strip(".f1.json").split("-")
        epoch, model_type, negative_sampler = info[0][len("epoch="):], info[1], info[2]
        with open(F1_PATH + f1_file, "r") as f:
            f1_json = json.load(f)
        f1 = f1_json["F1"]
        f1s["-".join([model_type, negative_sampler])][epoch] = f1
    return f1s


def impute_missing_f1s(f1s):
    
    tbox_random = f1s["tbox-random"]
    tbox_hierarchical = f1s["tbox-hierarchical"]
    vector_sim_random = f1s["vector_sim-random"]
    vector_sim_hierarchical = f1s["vector_sim-hierarchical"]

    imputed_f1s = defaultdict(list)
    for setting in f1s.keys():
        f1s_for_setting = []
        for epoch in range(0, 490, 40):
            f1s_for_setting.append(f1s[setting].get(str(epoch), 0))
        imputed_f1s[setting] = f1s_for_setting

    return dict(imputed_f1s)


def plot_f1s(f1s):

    x_values = list(range(0, 490, 40))
    
    colors=['blue', 'red', 'blue', 'red']
    markers=["s", "s", "+", "+"]

    for i in range(4):
        color = colors[i]
        marker = markers[i]

        plt.plot(x_values, f1s[i], color=color, marker=marker)

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("WordNet")

    # plt.legend(title=f'Average reduction in + edges: {red_pos}%\nAverage reduction in - edges: {red_neg}%\nAverage reduction in total edges: {red_tot}%')
    # plt.legend()

    plt.grid(True)
    plt.savefig("wordnet.png")
    plt.clf()


if __name__ == "__main__":
    
    f1s = collect_f1s()
    f1s = impute_missing_f1s(f1s)
    tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical = f1s["tbox-random"], f1s["tbox-hierarchical"], f1s["vector_sim-random"], f1s["vector_sim-hierarchical"]
    plot_f1s([tbox_random, tbox_hierarchical, vector_sim_random, vector_sim_hierarchical])
