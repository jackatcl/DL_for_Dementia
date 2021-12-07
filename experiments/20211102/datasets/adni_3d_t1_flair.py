import os, torch, pdb
import numpy as np
import json
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
import random 
import collections
from numpy import random as nprandom
import pickle
import glob
import re
import numpy as np
import pandas as pd
from random import shuffle
import random
import math
import nibabel as nib
from .augmentations import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ADNI_3D_T1_FLAIR(data.Dataset):

    # Removed dir_to_scan and dir_to_tsv
    # Include path and label tsv file
    def __init__(self, t1_path_label_file, flair_path_label_file, n_label = 3, percentage_usage = 1.0):
        if n_label == 3:
            LABEL_MAPPING = ["CN", "MCI", "AD"]
        elif n_label == 2:
            LABEL_MAPPING = ["CN", "AD"]
        self.LABEL_MAPPING = LABEL_MAPPING

        # This file includes Subject, Session, Label, Path
        # and path to the clinica step A pre-processed Space_T1w files
        t1_subject_tsv = pd.read_csv(t1_path_label_file, sep='\t').loc[:, ['Session', 'Label', 'Subject', 'Path']]
        t1_subject_tsv.rename({'Path' : 't1_path'}, axis=1, inplace=True)
        flair_subject_tsv = pd.read_csv(flair_path_label_file, sep='\t').loc[:, ['Session', 'Path']]
        flair_subject_tsv.rename({'Path' : 'flair_path'}, axis=1, inplace=True)
        self.subject_tsv = t1_subject_tsv.merge(flair_subject_tsv, how='left', on=['Session'])
        self.subject_tsv = self.subject_tsv[~self.subject_tsv.isna().any(axis=1)].drop_duplicates().reset_index(drop=True)

        self.subject_id = np.unique(self.subject_tsv.Subject.values)
        self.index_dic = dict(zip(self.subject_id, range(len(self.subject_id))))
        self.age_range = list(np.arange(0.0,120.0,0.5))

    def __len__(self):
        return len(self.subject_tsv)

    def __getitem__(self, idx):
        try:
            t1_path = self.subject_tsv.loc[idx, 't1_path']
            flair_path = self.subject_tsv.loc[idx, 'flair_path']

            if self.subject_tsv.iloc[idx].Label in [0, 1, 2]:
                label = self.subject_tsv.iloc[idx].Label
            else:
                print('WRONG LABEL VALUE!!!')
                label = -100
            mmse = 0  #self.subject_tsv.iloc[idx].mmse
            cdr_sub = 0  #self.subject_tsv.iloc[idx].cdr #cdr_sb #cdr#
            try:
                age = list(np.arange(0.0,120.0,0.5)).index(self.subject_tsv.iloc[idx].Age) #list(np.arange(0.0,25.0)).index(self.subject_tsv.iloc[idx].education_level)#
            except:
                age = []
                
            idx_out = self.index_dic[self.subject_tsv.iloc[idx].Subject]

            # load image, t1
            t1_image = nib.load(t1_path).get_data().squeeze()
            t1_image[np.isnan(t1_image)] = 0.0
            t1_image = (t1_image - t1_image.min())/(t1_image.max() - t1_image.min() + 1e-6)
            t1_image = np.expand_dims(t1_image,axis =0)
            t1_image = self.centerCrop(t1_image,96,96,96)

            # load image, flair
            flair_image = nib.load(flair_path).get_data().squeeze()
            flair_image[np.isnan(flair_image)] = 0.0
            flair_image = (flair_image - flair_image.min())/(flair_image.max() - flair_image.min() + 1e-6)
            flair_image = np.expand_dims(flair_image,axis =0)
            flair_image = self.centerCrop(flair_image,96,96,96)

        except Exception as e:
            print(f"Failed to load #{idx}: {flair_path}")
            print(f"Errors encountered: {e}")
            # print(traceback.format_exc())
            return None,None,None,None

        return t1_image.astype(np.float32), flair_image.astype(np.float32),label,idx_out,mmse,cdr_sub,age

    def centerCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height

        x = img.shape[1]//2 - length//2
        y = img.shape[2]//2 - width//2
        z = img.shape[3]//2 - height//2
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img

    def randomCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height
        x = random.randint(0, img.shape[1] - length)
        y = random.randint(0, img.shape[2] - width)
        z = random.randint(0, img.shape[3] - height )
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img
        
    def augment_image(self, image):
        sigma = np.random.uniform(0.0,1.0,1)[0]
        image = scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)
        return image

    def unpickling(self, path):
       file_return=pickle.load(open(path,'rb'))
       return file_return