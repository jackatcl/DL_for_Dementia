"""
    This cell is to create directories in accordance to BIDS structure
    from barlow_niigz to barlow_bids

    NOTE: The sub is actually sess. The sess is unknown.
    This file is deprecated.
"""

from glob import glob
import os
import pandas as pd
import numpy as np

input_dir = '/gpfs/data/razavianlab/data/mri/nyu/barlow_niigz/data/'
output_dir = '/gpfs/data/razavianlab/data/mri/nyu/barlow_bids/'

folders = glob(input_dir + '/*')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# else:
#     raise Exception('The root folder exists.')

id_summary = []
file_migration_summary = []
i = 0
for f_name in folders:
    if '_unmapped' in f_name: # skip unmapped folders
        continue

    # get patient id
    id = f_name.split('/')[-1]
    # create patient folders
    pat_folder_name = output_dir + 'sub-{}/'.format(id)
    if not os.path.exists(pat_folder_name):
        os.mkdir(pat_folder_name)

    # get all subdirectories
    sessions = [x[1] for x in os.walk(f_name)][0]
    # get a mapping from orig file location to destination file location
    for sess in sessions:
        temp = [x[2] for x in os.walk(os.path.join(f_name, sess))][0]
        for f in temp:
            orig_path = os.path.join(f_name, sess, f)
            dest_path = os.path.join(pat_folder_name, 'ses-{}/'.format(sess.split('-')[0]), 'anat', f)
            file_migration_summary.append(np.array([id, sess.split('-')[0], sess, orig_path, dest_path]))


    # create subdirectories for each session
    for sess in set([x.split('-')[0] for x in sessions]):
        ses_name = 'ses-{}/'.format(sess)
        sess_folder_name = pat_folder_name + ses_name
        if not os.path.exists(sess_folder_name):
            os.mkdir(sess_folder_name)
        if not os.path.exists(sess_folder_name + 'anat'):
            os.mkdir(sess_folder_name + 'anat')

    file_name = 'sub-{}_ses-{}'.format(id, sess)
    id_summary.append(np.array(['sub-{}'.format(id), 'ses-{}'.format(sess)]))

id_summary_df = pd.DataFrame(np.array(id_summary), columns=['participant_id', 'session_id'])
id_summary_df.to_csv(output_dir + 'participant_table.tsv', index=False, sep='\t')

file_migration_df = pd.DataFrame(np.array(file_migration_summary), columns=['participant_id', 'session_id', 'sess_info', 'orig_path', 'dest_path'])
file_migration_df.to_csv('/gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/' + 'file_migration_table_v1.tsv', index=False, sep='\t')
