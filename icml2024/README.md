# Experiment Pipeline

## Precompute and cache negatives for hierarchy-aware negative sampling

Run the script to precompute and cache the matrix of tails to negative heads for all the transitively-closed graphs:

```
cd ./0_compute_and_cache_negatives
sbatch --array=1-13 ./compute_and_cache_negatives.sh
```

## Run vector_sim hyperparameter tuning for per-graph best learning rate and negative weight

First, create the sweeps. Specify `output_sweep_ids_file` to the text file where the sweep ids for the created sweeps get stored, one per line. In total, 26 sweeps will get created, for each of the 13 graphs times `negative_ratio=[4, 128]`:

```
cd ./1_vector_sim_hyperparameter_tuning/
python3 create_sweeps.py --output_sweep_ids_file ./sweep_ids.txt
```

Now launch agents to run the sweeps that have been created, in [Array Batch jobs](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/arrays/) mode:

```
sbatch --array=1-26 ./launch_agents.sh
```

Finally, aggregate the best learning rate and negative weight from the sweeps and store in a json file, to be retrieved by the `vector_sim` sweeps for the following experiments:

```
python3 query_wandb_for_best_learning_rate_and_negative_weight_per_graph_type.py \
--input_sweep_ids_file ./sweep_ids.txt \
--output_json_file ./graph_type_to_best_learning_rate_and_negative_weight.json
```

This file gets utilized by the `train_vector_sim` entrypoint, which loads it and retrieves the corresponding learning rate and negative weight to use, depending on the run's hyperparameters.

## Run tbox experiments

Kick off the sweep for all `tbox` experiments:

```
python3 create_sweep.py --model tbox
```

## Run vector_sim experiments

Kick off the sweep for all `vector_sim` experiments:

```
python3 create_sweep.py --model tbox --lr_nw_json ../1_vector_sim_hyperparameter_tuning/graph_type_to_best_learning_rate_and_negative_weight.json
```