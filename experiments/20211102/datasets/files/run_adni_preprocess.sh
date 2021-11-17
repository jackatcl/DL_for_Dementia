#!/bin/bash
#SBATCH --job-name=adni_preprocess
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12-24:00:00
#SBATCH --output=slurm_%j.out

module load anaconda3

module load matlab
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export MATLAB_HOME=/gpfs/share/apps/matlab/current/bin/
export PATH=${MATLAB_HOME}:${PATH}
export MATLABCMD=${MATLAB_HOME}/matlab
conda activate /gpfs/data/razavianlab/capstone/2021_dementia/env_ben/miniconda3/envs/clinicaEnv
export FREESURFER_HOME=/gpfs/share/apps/freesurfer/6.0.0/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SPM_HOME= /gpfs/data/razavianlab/capstone/2021_dementia/software/spm12/spm12
clinica run t1-volume-tissue-segmentation '/gpfs/data/razavianlab/data/mri/nyu/barlow_niigz/data/100027089657/2499308044-20120717-AX_3D_MPR-6' '/gpfs/data/razavianlab/capstone/2021_dementia/dataset'
