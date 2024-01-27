#!/bin/bash
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH -t 7:00:00
#SBATCH -o /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/3_wordnet/slurm/slurm-%j.out

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/3_wordnet/

PARTITION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "./partition_assignments.txt")
echo ${PARTITION}
scontrol update PartitionName=${PARTITION}

MODEL_CKPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "./model_ckpts.txt")
MODEL_TYPE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "./model_types.txt")

/usr/bin/env python /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/scripts/box-training-methods wordnet_full_eval --model_type=$MODEL_TYPE --model_checkpoint=$MODEL_CKPT
