#!/bin/bash
#SBATCH --job-name=train_tdex
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=learnlab
#SBATCH --signal=SIGUSR1@120
#SBATCH --time=02-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=akashsharma02@meta.com

source /data/home/akashsharma02/miniforge3/etc/profile.d/conda.sh
conda activate tactile_dexterity

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun python train.py experiment=tdex-run1 hydra.job.id=$SLURM_JOB_ID $1 $2 $3 $4 $5
