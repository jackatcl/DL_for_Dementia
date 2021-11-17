import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--orig_file",
                    type=str,
                    required=True,
                    help="original label file")

arguments = parser.parse_args()

orig_file_df = pd.read_csv(arguments.orig_file, sep='\t', index_col=0)

subjects = list(set(orig_file_df.Subject.to_list()))
subjects = np.random.RandomState(seed=42).permutation(subjects)

train_subject = subjects[:int(len(subjects) * .7)]
val_subject = subjects[int(len(subjects) * .7): int(len(subjects) * .85)]
test_subject = subjects[int(len(subjects) * .85):]

train_df = orig_file_df[orig_file_df.Subject.isin(train_subject)]
val_df = orig_file_df[orig_file_df.Subject.isin(val_subject)]
test_df = orig_file_df[orig_file_df.Subject.isin(test_subject)]

print('Train shape: {}, val shape: {}, test shape: {}'.format(train_df.shape, val_df.shape, test_df.shape))

orig_file_name = arguments.orig_file.split('.')[0]
train_df.to_csv(orig_file_name + '_train.tsv', sep='\t', index=False)
val_df.to_csv(orig_file_name + '_val.tsv', sep='\t', index=False)
test_df.to_csv(orig_file_name + '_test.tsv', sep='\t', index=False)