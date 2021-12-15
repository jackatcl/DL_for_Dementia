import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
from numpy.random import permutation
import yaml


"""
    Utility Functions
"""

def evaluation_models(model_name,data_loader, expansion_list = [8],  num_trails = 10,percentage = 1, use_age = False, norm_type= 'Instance'):
    for i,ep in enumerate(expansion_list):
        cfg['model']['expansion'] = ep
        cfg['model']['norm_type'] = norm_type
        model = build_model(cfg)
        #TODO: Change model directory
        best_model_dir = '/gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/experiments/20211102/saved_model/'
        pretrained_dict = torch.load(best_model_dir+model_name + '_model_low_loss.pth.tar',map_location='cpu')['state_dict']
        old_ks = list(pretrained_dict.keys()).copy()
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:]in model_dict.keys())}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        model = model.to(device)
        model = model.eval()

        #TODO:
    return all_acc, all_balanced_acc, all_auc


"""
    End Utility Functions
"""

if __name__ == '__main__':
    
    # torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load config
    config_path = '/gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/experiments/20211102/configs/config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.load(f)

    # load data
    from datasets.adni_3d import ADNI_3D
    from models.build_model import build_model


    # path to tsv file containing pre-processed image file path and label
    path_to_tsv = '/gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/experiments/20211102/label_and_file_path/20211206/flair_test.tsv'

    Test_dataset = ADNI_3D(path_label_file=path_to_tsv,  n_label = cfg['model']['n_label'])
    Test_loader = torch.utils.data.DataLoader(
            Test_dataset, batch_size=cfg['data']['val_batch_size'], shuffle=False,
            num_workers=cfg['data']['workers'], pin_memory=True)

    category = ['CN','MCI','AD']

    # Load model
    model = build_model(cfg)
    # /gpfs/home/lc3424/capstone/2021_dementia/lc3424_workspace/experiments/20211102/saved_model/volume_retrain_flair_2/volume_retrain_flair_train_perc_100.0_expansion_8_model_low_loss.pth.tar
    model_file_name = 'volume_retrain_flair_2/volume_retrain_flair_train_perc_100.0_expansion_8'
    # model_file_name = 'volume_retrain_train_perc_100.0_expansion_0'

    
