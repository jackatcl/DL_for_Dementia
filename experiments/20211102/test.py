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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_name = 'config.yaml'
with open(os.path.join('./'+config_name), 'r') as f:
    cfg = yaml.load(f)

from datasets.adni_3d import ADNI_3D
from models.build_model import build_model

model = build_model(cfg)
model_file_name = 'age_expansion_8'

from torchsummary import summary

print(summary(model, (1, 96, 96, 96)))