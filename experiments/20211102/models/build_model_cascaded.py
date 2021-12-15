import json
import pickle
import timeit
import sys
import numpy as np
import torch

from .classifier import LinearClassifierAlexNet
from .models import NetWork
from .LinearModel import alex_net_complete, alex_net_cascaded_complete


def prepare_model_cascaded(arch = 'ours', in_channel=1, 
    feat_dim=128, n_hid_main=256, n_label=3, out_dim = 128, 
    expansion = 8, type_name='conv3x3x3', norm_type = 'Instance'):

    if arch == 'ours':
        t1_image_embeding_model = NetWork(in_channel=in_channel,feat_dim=feat_dim, expansion = expansion, type_name=type_name, norm_type=norm_type)
        flair_image_embeding_model = NetWork(in_channel=in_channel,feat_dim=feat_dim, expansion = expansion, type_name=type_name, norm_type=norm_type)

    else:
        print('Wrong Archetecture!')
    # generate the classifier
    classifier = LinearClassifierAlexNet(in_dim=feat_dim * 2, n_hid=n_hid_main, n_label=n_label)
    main_model = alex_net_cascaded_complete(t1_image_embeding_model, flair_image_embeding_model, classifier)

    return main_model

def build_model_cascaded(config, input_dim = 3):
    arch = config['model']['arch']
    in_channel = config['model']['input_channel']
    feat_dim = config['model']['feature_dim']
    n_label = config['model']['n_label']
    n_hid_main = config['model']['nhid']
    out_dim = config['adv_model']['out_dim']
    expansion = config['model']['expansion']
    type_name = config['model']['type_name']
    norm_type = config['model']['norm_type']

    main_model = prepare_model_cascaded(arch = arch, in_channel=in_channel, feat_dim=feat_dim, 
        n_hid_main=n_hid_main,  n_label=n_label, out_dim = out_dim, 
        expansion= expansion, type_name=type_name, norm_type=norm_type)

    # if config['training_parameters']['pretrain'] is not None:
    #     pretrained_dict = torch.load(config['training_parameters']['pretrain'], map_location='cpu')['state_dict']
    #     model_dict = main_model.state_dict()
    #     pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:]in model_dict.keys())}
    #     model_dict.update(pretrained_dict) 
    #     print(model_dict.keys())
    #     main_model.load_state_dict(model_dict)
    #     print('pre-trained model loaded')
    return main_model