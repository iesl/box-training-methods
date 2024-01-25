#!/bin/bash
#SBATCH -c 16
#SBATCH -p gypsum-rtx8000
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/3_wordnet/slurm/slurm-%j.out

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/3_wordnet/

SWEEP_ID=$1
wandb agent -e hierarchical-negative-sampling -p icml2024 --count 1 ${SWEEP_ID}
