if __name__ == "__main__":

    model_ckpts = []
    model_types = []

    for epoch in list(range(0, 491, 40)) + list(range(20, 491, 40)) + list(range(10, 491, 20)):     # 13, 12, 25
        for model_type in ["vector_sim", "tbox"]:
            for negative_sampler in ["random", "hierarchical"]:
                model_ckpts.append(f"/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/icml2024_wordnet_models/wordnet_full.epoch={epoch}-{model_type}-{negative_sampler}.pt")
                model_types.append(model_type)

    with open("./model_ckpts.txt", "w") as f:
        f.write("\n".join(model_ckpts))
    with open("./model_types.txt", "w") as f:
        f.write("\n".join(model_types))
