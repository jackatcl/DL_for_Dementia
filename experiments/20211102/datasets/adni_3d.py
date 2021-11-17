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

class ADNI_3D(data.Dataset):

    # Removed dir_to_scan and dir_to_tsv
    # Include path and label tsv file
    def __init__(self, path_label_file, n_label = 3, percentage_usage = 1.0):
        if n_label == 3:
            LABEL_MAPPING = ["CN", "MCI", "AD"]
        elif n_label == 2:
            LABEL_MAPPING = ["CN", "AD"]
        self.LABEL_MAPPING = LABEL_MAPPING    

        # This file includes Subject, Session, Label, Path
        # and path to the clinica step A pre-processed Space_T1w files
        self.subject_tsv = pd.read_csv(path_label_file, sep='\t') 
        self.subject_id = np.unique(self.subject_tsv.Subject.values)
        self.index_dic = dict(zip(self.subject_id, range(len(self.subject_id))))
        self.age_range = list(np.arange(0.0,120.0,0.5))

    def __len__(self):
        return len(self.subject_tsv)

    def __getitem__(self, idx):
        try:
            path = self.subject_tsv.loc[idx, 'Path']
        
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

            # load image
            image = nib.load(path).get_data().squeeze()
            image[np.isnan(image)] = 0.0
            image = (image - image.min())/(image.max() - image.min() + 1e-6)
            image = np.expand_dims(image,axis =0)

            image = self.centerCrop(image,96,96,96)

        except Exception as e:
            print(f"Failed to load #{idx}: {path}")
            print(f"Errors encountered: {e}")
            print(traceback.format_exc())
            return None,None,None,None

        return image.astype(np.float32),label,idx_out,mmse,cdr_sub,age

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