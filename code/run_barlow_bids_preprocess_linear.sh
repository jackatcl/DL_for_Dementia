#!/bin/bash
#SBATCH --job-name=adni_preprocess_linear
#SBATCH --ntasks=12
#SBATCH --mem-per-cpu=16G
#SBATCH --time=14-24:00:00
#SBATCH --output=slurm_%j.out

module load anaconda3

module load matlab/R2018a
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export ANTSPATH="/gpfs/home/lc3424/capstone/2021_dementia/software/install/bin"
export PATH=${ANTSPATH}:${PATH}
export MATLAB_HOME=/gpfs/share/apps/matlab/R2018a/bin
export PATH=${MATLAB_HOME}:${PATH}
export MATLABCMD=${MATLAB_HOME}/matlab
source /gpfs/data/razavianlab/capstone/2021_dementia/env_ben/miniconda3/bin/activate
conda activate clinicaEnv
export FREESURFER_HOME=/gpfs/share/apps/freesurfer/6.0.0/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SPM_HOME=/gpfs/data/razavianlab/skynet/alzheimers/spm12
matlab -nodesktop -nosplash -singleCompThread -r "addpath('/gpfs/data/razavianlab/skynet/alzheimers/spm12');exit"
clinica run t1-linear '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids_t1_unprocessed_linear' '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids_t1_preprocessed_lin' -tsv '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids_t1_unprocessed_linear/participant_table.tsv' -wd '/gpfs/data/razavianlab/data/mri/nyu/WD_barlow_bids_t1_preprocess_lin_part_a'  -np 24
