#!/bin/bash
#SBATCH -c 16  # Number of Cores per Task
#SBATCH --mem=128GG  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o slurm/slurm-%j.out  # %j = job ID

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/0_compute_and_cache_negatives/

PYTHONPATH=/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods \
python ./wordnet.py