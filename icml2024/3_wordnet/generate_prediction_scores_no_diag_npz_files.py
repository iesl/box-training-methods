import os


if __name__ == "__main__":
    
    npzs = sorted(os.listdir("/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/icml2024_wordnet_prediction_scores_no_diag/"), key=lambda c: int(c.split('-')[0][len("epoch="):]))
    npzs = ["/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/icml2024_wordnet_prediction_scores_no_diag/" + c for c in npzs]

    with open("./prediction_scores_no_diag_npzs.txt", "w") as f:
        f.write("\n".join(npzs))
