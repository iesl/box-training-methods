#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH --mem=512G  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o slurm/f1/slurm-%j.out  # %j = job ID

source /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/anaconda3/bin/activate box-training-methods
cd /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/icml2024/3_wordnet/

PREDITION_SCORES_NO_DIAG_NPY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "./prediction_scores_no_diag_npys.txt")

PYTHONPATH=/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods \
python ./calculate_f1.py --prediction_scores_no_diag_npy=$PREDITION_SCORES_NO_DIAG_NPY