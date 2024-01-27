import os


if __name__ == "__main__":
    
    model_ckpts = sorted(os.listdir("/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/icml2024_wordnet_models/"), key=lambda c: int(c.split('-')[0][len("wordnet_full.epoch="):]))
    model_types = [c.split('-')[-2] for c in model_ckpts]
    model_ckpts = ["/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/icml2024_wordnet_models/" + c for c in model_ckpts]

    with open("./model_ckpts.txt", "w") as f:
        f.write("\n".join(model_ckpts))
    with open("./model_types.txt", "w") as f:
        f.write("\n".join(model_types))
