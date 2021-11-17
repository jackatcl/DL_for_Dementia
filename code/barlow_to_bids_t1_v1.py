from glob import glob
import os
import pandas as pd
import numpy as np
from shutil import copyfile
from collections import defaultdict


input_dir = '/gpfs/data/razavianlab/data/mri/nyu/barlow_niigz/data/'
output_dir = '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids_t1_unprocessed_volume_2/'

file_summary = pd.read_csv('/gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/code/t1_file_path_with_label.tsv', sep='\t')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    raise Exception('The root folder exists.')

id_summary = set()
historical_sess = defaultdict(int)

for _, row in file_summary.iterrows():
    if row.dest_path.replace('.nii.gz', '')[-1].isalpha():
        continue
    # get patient id
    id = row.Subject
    # create patient folders
    pat_folder_name = output_dir + 'sub-{}/'.format(id)
    if not os.path.exists(pat_folder_name):
        os.mkdir(pat_folder_name)

    # create session folder
    sess = int(row.Session)
    historical_sess[sess] += 1
    if historical_sess[sess] > 1:
        sess = str(sess) + '_{}'.format(historical_sess[sess])
    
    ses_name = 'ses-{}/'.format(sess)
    sess_folder_name = pat_folder_name + ses_name
    if not os.path.exists(sess_folder_name):
        os.mkdir(sess_folder_name)
    if not os.path.exists(sess_folder_name + 'anat'):
        os.mkdir(sess_folder_name + 'anat')

    cur_file_name = row.dest_path.replace('.nii.gz', '_T1w.nii.gz')
    file_name = 'sub-{}_ses-{}_'.format(id, sess) + cur_file_name
    # copy file
    copyfile(row.Path, sess_folder_name + 'anat/' + file_name)

    id_summary.add(('sub-{}'.format(id), 'ses-{}'.format(sess)))

id_summary_df = pd.DataFrame(np.array(list(id_summary)), columns=['participant_id', 'session_id'])
id_summary_df.to_csv(output_dir + 'participant_table.tsv', index=False, sep='\t')
