import wandb


def main():

    sweep_config = {
    "command":[
        "${env}",
        "${interpreter}",
        "${program}",
        f"wordnet_full",
        "${args}",
    ],
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "[Eval] F1"
    },
    "name": "wordnet_full",
    "parameters": {
        "model_type": {
            "values": ["tbox", "vector_sim"]
        },
        "negative_sampler": {
            "values": ["hierarchical", "random"]
        },
    },
    "program": "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/scripts/box-training-methods"
    }

    sweep_id = wandb.sweep(sweep=sweep_config, entity="hierarchical-negative-sampling", project="icml2024")
    print(sweep_id)


if __name__ == "__main__":
    
    main()
