#!/bin/bash
#SBATCH --job-name=direct_inf
#SBATCH --partition=gpu4_dev
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --time=4:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1

module load cuda90/toolkit/9.1.176
module load miniconda3

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi
 
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

source /gpfs/data/razavianlab/capstone/2021_dementia/env_ben/miniconda3/bin/activate
conda activate env_gpu_py36

conda run -n env_gpu_py36 python3 /gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/experiments/20211102/model_eval_cascaded.py