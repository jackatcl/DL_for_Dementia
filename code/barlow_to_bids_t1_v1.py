from glob import glob
import os
import pandas as pd
import numpy as np
from shutil import copyfile
from collections import defaultdict


input_dir = '/gpfs/data/razavianlab/data/mri/nyu/barlow_niigz/data/'
output_dir = '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids_t1_unprocessed_linear/'

file_summary = pd.read_csv('/gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/file_migration_table_t1_v4.tsv', sep='\t')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    raise Exception('The root folder exists.')

id_summary = set()
historical_sess = defaultdict(int)
historical_sess_info = defaultdict(int)

for _, row in file_summary.iterrows():
    # get patient id
    id = row.random_pat_id
    # create patient folders
    pat_folder_name = output_dir + 'sub-{}/'.format(id)
    if not os.path.exists(pat_folder_name):
        os.mkdir(pat_folder_name)

    # create session folder
    sess = int(row['de-identified acc'])
    if row.sess_info not in historical_sess_info:
        historical_sess[sess] += 1
        historical_sess_info[row.sess_info] = historical_sess[sess]
    if historical_sess[sess] > 1:
        sess = str(sess) + '_{}'.format(historical_sess_info[row.sess_info])
    
    ses_name = 'ses-{}/'.format(sess)
    sess_folder_name = pat_folder_name + ses_name
    if not os.path.exists(sess_folder_name):
        os.mkdir(sess_folder_name)
    if not os.path.exists(sess_folder_name + 'anat'):
        os.mkdir(sess_folder_name + 'anat')

    cur_file_name = row.dest_path.replace('.nii.gz', '_T1w.nii.gz').replace('.json', '_T1w.json')
    file_name = 'sub-{}_ses-{}_'.format(id, sess) + cur_file_name
    # copy file
    copyfile(row.orig_path, sess_folder_name + 'anat/' + file_name)

    id_summary.add(('sub-{}'.format(id), 'ses-{}'.format(sess)))

id_summary_df = pd.DataFrame(np.array(list(id_summary)), columns=['participant_id', 'session_id'])
id_summary_df.to_csv(output_dir + 'participant_table.tsv', index=False, sep='\t')
