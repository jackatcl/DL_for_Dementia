#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=14-24:00:00
#SBATCH --output=slurm_%j.out

module load anaconda3

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
##source /gpfs/scratch/sl5924/env/bin/activate
python3 /gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/code/barlow_to_bids_t1_v1.py