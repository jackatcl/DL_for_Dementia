#!/bin/bash
#SBATCH --job-name=flair_reg
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=4-24:00:00
#SBATCH --output=slurm_%j.out

module load miniconda3
module load freesurfer/6.0.0
module load fsl/6.0.0

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
# matlab -nodesktop -nosplash -singleCompThread -r "addpath('/gpfs/data/razavianlab/skynet/alzheimers/spm12');exit"

conda run -n clinicaEnv python flair_registration.py