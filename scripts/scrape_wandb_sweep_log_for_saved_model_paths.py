import os, re, json
from collections import defaultdict


run_re = re.compile(r"run-\d{8}_\d{6}-([a-z0-9]{8})")


def scrape_log(log_path):
    
    MODEL_MARKER = "Saving model as "
    PREDICTION_MARKER = "Saving predictions to: "
    PREDICTION_SCORES_MARKER = "Saving prediction_scores to: "
    METRICS_MARKER = "Saving metrics to: "
    
    with open(log_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    
    model_paths = [l for l in lines if MODEL_MARKER in l]
    model_paths = [p[p.index(MODEL_MARKER) + len(MODEL_MARKER):].strip("'") for p in model_paths]
    
    hashes = [run_re.search(p).groups()[0] for p in model_paths]

    prediction_paths = [l for l in lines if PREDICTION_MARKER in l]
    prediction_paths = [p[p.index(PREDICTION_MARKER) + len(PREDICTION_MARKER):] for p in prediction_paths]

    prediction_scores_paths = [l for l in lines if PREDICTION_SCORES_MARKER in l]
    prediction_scores_paths = [p[p.index(PREDICTION_SCORES_MARKER) + len(PREDICTION_SCORES_MARKER):] for p in prediction_scores_paths]

    metrics_paths = [l for l in lines if METRICS_MARKER in l]
    metrics_paths = [p[p.index(METRICS_MARKER) + len(METRICS_MARKER):] for p in metrics_paths]

    hash_to_data_sorted_by_epoch = defaultdict(lambda: defaultdict(list))
    for i, model_path in enumerate(model_paths):
        hash_to_data_sorted_by_epoch[hashes[i]]['models'].append(model_path)
        hash_to_data_sorted_by_epoch[hashes[i]]['predictions'].append(prediction_paths[i])
        hash_to_data_sorted_by_epoch[hashes[i]]['prediction_scores'].append(prediction_scores_paths[i])
        hash_to_data_sorted_by_epoch[hashes[i]]['metrics'].append(metrics_paths[i])

    return hash_to_data_sorted_by_epoch


if __name__ == "__main__":

    r_sweep_name = "hns.random.tc.save"
    r_hash_to_paths = scrape_log(log_path="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/logs/wandb/hierarchical-negative-sampling/hns/ch4uumvm/2023-05-01-14-44-55-773678062/log.err")
    with open(f"./saved_model_paths/{r_sweep_name}.json", "w") as f:
        json.dump(r_hash_to_paths, f, indent=4, sort_keys=False)

    h_e_sweep_name = "hns.hierarchical.exact.tc.save"
    h_e_hash_to_paths = scrape_log(log_path="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/logs/wandb/hierarchical-negative-sampling/hns/ff7ym8vk/2023-05-01-14-45-29-523391763/log.err")
    with open(f"./saved_model_paths/{h_e_sweep_name}.json", "w") as f:
        json.dump(h_e_hash_to_paths, f, indent=4, sort_keys=False)

    h_u_d_sweep_name = "hns.hierarchical.sampled.tc.save"
    h_u_d_hash_to_paths = scrape_log(log_path="/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/logs/wandb/hierarchical-negative-sampling/hns/0bpboyks/2023-05-01-14-46-29-740870604/log.err")
    with open(f"./saved_model_paths/{h_u_d_sweep_name}.json", "w") as f:
        json.dump(h_u_d_hash_to_paths, f, indent=4, sort_keys=False)
