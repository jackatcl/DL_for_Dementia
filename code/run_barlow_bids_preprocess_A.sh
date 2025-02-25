#!/bin/bash
#SBATCH --job-name=pre_vol
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=14-24:00:00
#SBATCH --output=slurm_%j.out

module load miniconda3

module load matlab/R2018a
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export MATLAB_HOME=/gpfs/share/apps/matlab/R2018a/bin
export PATH=${MATLAB_HOME}:${PATH}
export MATLABCMD=${MATLAB_HOME}/matlab
source /gpfs/data/razavianlab/capstone/2021_dementia/env_ben/miniconda3/bin/activate
conda activate clinicaEnv
export FREESURFER_HOME=/gpfs/share/apps/freesurfer/6.0.0/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SPM_HOME=/gpfs/data/razavianlab/skynet/alzheimers/spm12
matlab -nodesktop -nosplash -singleCompThread -r "addpath('/gpfs/data/razavianlab/skynet/alzheimers/spm12');exit"
clinica run t1-volume-tissue-segmentation '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids_t1_unprocessed_volume_2_centered' '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids_t1_preprocess_A_run_2' -np 8 -tsv '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids_t1_unprocessed_volume_2_centered/participant_table.tsv' -wd '/gpfs/data/razavianlab/data/mri/nyu/WD_barlow_bids_t1_preprocess_A_run_2' 