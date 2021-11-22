"""
    Register FLAIR images with reference to t1 scans from the same scan session using FSL FLIRT
"""

import pandas as pd
import os
import subprocess
from nipype.interfaces.fsl.utils import Reorient2Std
import timeit


start = timeit.timeit()
out_dir = '/gpfs/data/razavianlab/data/mri/nyu/barlow_flair_registered'
os.chdir('/gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/code')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# T1-FLAIR file pairs
t1_flair_file = pd.read_csv('t1_flair_file_match.tsv', sep='\t')

for i in range(t1_flair_file.shape[0]):
    print(i)
    if i % 100 == 0:
        end = timeit.timeit()
        print('{} scans processed with time elapsed {}s'.format(i, end - start))
        with open('flair_registration_log.txt', 'a') as the_file:
            the_file.write('{} scans processed with time elapsed {}s\n'.format(i, end - start))

    temp = t1_flair_file.loc[i, ['flair_path', 't1_path']]
    flair, t1 = temp.flair_path, temp.t1_path

    try:
        # Reorient
        reorient = Reorient2Std()
        # Flair reorient
        reorient.inputs.in_file = flair
        reorient.inputs.out_file = 'temp/flair_reorient.nii.gz'
        reorient.inputs.output_type = 'NIFTI_GZ'
        res = reorient.run()

        # t1 reorient
        reorient.inputs.in_file = t1
        reorient.inputs.out_file = 'temp/t1_reorient.nii.gz'
        reorient.inputs.output_type = 'NIFTI_GZ'
        res = reorient.run()

        # FLIRT registration
        out_file = flair.split('/')[-1].replace('.nii.gz', '_flair_reg.nii.gz')
        out_file_path = os.path.join(out_dir, out_file)
        mat_file_path = os.path.join(out_dir, out_file.replace('.nii.gz', 'transformation.mat'))
        t1_flair_file.loc[i, 'flair_registered_file_path'] = out_file_path
        subprocess.call(['flirt', '-in', 'temp/t1_reorient.nii.gz', '-ref', 'temp/flair_reorient.nii.gz', '-out', out_file_path, '-omat', mat_file_path, '-dof', '6'])
    except Exception:
        print('Error processing FLIRT with:\nT1 file: {}\nand\nFLAIR file: {}'.format(t1, flair))
        print(Exception)
        with open('flair_registration_log.txt', 'a') as the_file:
            the_file.write('Error processing FLIRT with: T1 file: {} and FLAIR file: {}\n'.format(t1, flair))

t1_flair_file.to_csv('t1_flair_file_match_with_reg.tsv', sep='\t', index=False)