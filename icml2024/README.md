# Experiment Pipeline

## Precompute and cache negatives for hierarchy-aware negative sampling

Run the script to precompute and cache the matrix of tails to negative heads for all the transitively-closed graphs (this will also compute the stats and store them as a json):

```
cd ./0_compute_and_cache_negatives
sbatch --array=1-94 ./synthetic_graphs.sh
sbatch ./wordnet.sh
```

Go to the next directory for tuning `learning_rate` and `negative_weight`:

```
cd ../1_hyperparameter_tuning
```

## Run hyperparameter tuning for best learning rate and negative weight

First, create the sweeps. Specify `output_sweep_ids_file` to the text file where the sweep ids for the created sweeps get stored, one per line. In total, 208 sweeps will get created, for each of the 13 graphs times `model_type=["vector_sim", "tbox]`, `negative_sampler=["random", "hierarchical"]`, `negative_ratio=[4, 128]`, `sample_positive_edges_from_tr_or_tc=["tr", "tc"]`:

```
cd ./1_hyperparameter_tuning/
python3 create_sweeps.py --output_sweep_ids_file ./sweep_ids.txt
```

Now launch agents to run the sweeps that have been created, in [Array Batch jobs](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/arrays/) mode:

```
sbatch --array=1-208 ./launch_agents.sh
```

Finally, aggregate the best learning rate and negative weight from the sweeps and store in a json file, to be retrieved by the `synthetic_graphs` sweeps for the following experiments:

```
python3 query_wandb_for_best_learning_rate_and_negative_weight_per_graph_type.py \
--input_sweep_ids_file ./sweep_ids.txt \
--output_json_file ./configuration_to_best_learning_rate_and_negative_weight.json
```

This file gets utilized by the `synthetic_graphs` entrypoint, which loads it and retrieves the corresponding learning rate and negative weight to use, depending on the run's other hyperparameters.

Finally, change to the next directory for running the experiments on synthetic graphs:

```
cd ../2_synthetic_graphs
```

## Run tbox synthetic graphs experiments

Create the sweep for all synthetic graphs experiments:

```
python3 create_sweep.py --lr_nw_json ../1_hyperparameter_tuning/ configuration_to_best_learning_rate_and_negative_weight.json
```

This will print out the `SWEEP_ID` of the created sweep. Launch the agents for the sweep as follows:

```
sbatch --array=1-118 ./launch_agents.sh SWEEP_ID
```

Note that I have created `./partition_assignments.txt` (used by `./launch_agents.sh` in array mode) to utilize all of the gpus allowed for one PhD student as per [Unity documentation](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/gypsum/) (i.e. 12 for `gypsum-m40`, 40 for `gypsum-titanx`, etc.).

There are a total of `94 * 2 * 2 * 2 * 2 = 1504` runs in this grid search, dividing by `118` agents gives `>12` runs per agent, and I set `--count 25` just to account for imbalanced completion times for different runs.

```
cd ../3_wordnet
```