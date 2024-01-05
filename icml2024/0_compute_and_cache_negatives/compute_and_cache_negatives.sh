#!/bin/bash
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o slurm/slurm-%j.out  # %j = job ID

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/0_compute_and_cache_negatives/

GRAPH_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "./graph_dir_paths.txt")

PYTHONPATH=/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods \
python ./compute_and_cache_negatives.py \
--graph_dir=${GRAPH_DIR}