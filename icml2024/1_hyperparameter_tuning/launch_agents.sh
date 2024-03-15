#!/bin/bash
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/1_hyperparameter_tuning/slurm/slurm-%j.out

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/1_hyperparameter_tuning/

SWEEP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "./sweep_ids.txt")
PARTITION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "./sweep_partition_assignments.txt")
echo ${PARTITION}
scontrol update PartitionName=${PARTITION}

wandb agent -e hierarchical-negative-sampling -p icml2024 --count 25 ${SWEEP_ID}
