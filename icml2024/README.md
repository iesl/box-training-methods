# Experiment Pipeline

## Precompute and cache negatives for hierarchy-aware negative sampling

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

## Run tbox experiments

## Run vector_sim experiments
