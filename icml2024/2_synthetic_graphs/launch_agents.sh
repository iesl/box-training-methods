#!/bin/bash
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH -t 10:00:00
#SBATCH -o /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/2_synthetic_graphs/slurm/v2/slurm-%j.out

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/2_synthetic_graphs/

SWEEP_ID=$1
PARTITION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "./partition_assignments.txt")
echo ${PARTITION}
scontrol update PartitionName=${PARTITION}

wandb agent -e hierarchical-negative-sampling -p icml2024 --count 25 ${SWEEP_ID}
