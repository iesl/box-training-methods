#!/bin/bash
#SBATCH -c 4
#SBATCH -p gypsum-2080ti
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -o /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/3_wordnet/slurm/slurm-%j.out

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/3_wordnet/

SWEEP_ID=$1
wandb agent -e hierarchical-negative-sampling -p icml2024 --count 1 ${SWEEP_ID}
