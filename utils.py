from __future__ import print_function
import os
import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
import importlib
import sys
import pickle
import torchvision
import matplotlib.pyplot as plt
from DeepDIGCode import config
args = config.args

_classes = args.classes.split(';')
_classes = [int(a) for a in _classes]
classes={'s':_classes[0],'t':_classes[1]}
dataset_utils = importlib.import_module("DeepDIG.PreTrainedModels." + args.dataset + ".utils")
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'/PreTrainedModels/'+args.dataset+'/'+ args.pre_trained_model)
sys.path.append(dir_path+'/PreTrainedModels/'+args.dataset)

ae_module = importlib.import_module("DeepDIG.PreTrainedModels." + args.dataset + ".ae")
if not os.path.exists(args.project_dir+'PreTrainedModels/'+args.dataset+'/'+ args.pre_trained_model + '/({},{})'.format(classes['s'], classes['t'])):
    os.makedirs(args.project_dir+'PreTrainedModels/'+args.dataset+'/'+ args.pre_trained_model + '/({},{})'.format(classes['s'], classes['t']))

working_dir = args.project_dir+'PreTrainedModels/'+args.dataset+'/'+ args.pre_trained_model + '/({},{})'.format(classes['s'], classes['t'])+'/'
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

class_s_t_dir = working_dir+str(classes['s']) + '_' + str(classes['t']) +'/'
if not os.path.exists(class_s_t_dir):
    os.makedirs(class_s_t_dir)

class_t_s_dir = working_dir+str(classes['t']) + '_' + str(classes['s']) +'/'
if not os.path.exists(class_t_s_dir):
    os.makedirs(class_t_s_dir)

cuda_enabled = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda_enabled else "cpu")

def get_pre_trained_model_input_shape(batch_size = args.batch_size):
    pre_trained_model_input_shape = []
    pre_trained_model_input_shape.append(batch_size)
    for x in args.pre_trained_model_input_shape.split(';'):
        pre_trained_model_input_shape.append(int(x))
    return pre_trained_model_input_shape

reconst_loss_function = torch.nn.BCELoss()
BCELoss = torch.nn.BCELoss()

def loss_function_ae(reconst_ae, data, pre_trained_model_reconst_probs,
                     pre_trained_model_data_probs, batch_size, target, device):
    prediction_pre_trained_model = pre_trained_model_data_probs.max(1, keepdim=True)[1].cpu().numpy().reshape(batch_size)
    reconstrtion_loss = reconst_loss_function(reconst_ae, data)

    attacked_targets = torch.zeros(pre_trained_model_data_probs.shape)
    for i, j in zip(range(attacked_targets.shape[0]), prediction_pre_trained_model):
        attacked_targets[i][target] = 1

    attacked_targets = attacked_targets.to(device)

    targte_loss = BCELoss(pre_trained_model_reconst_probs, attacked_targets)
    return reconstrtion_loss  + args.alpha * targte_loss


def get_pre_trained_model_features_predictions(data,pre_trained_model,get_probs=False):
    features = []
    predictions = []
    probs=[]
    with torch.no_grad():
        for i in range(0,len(data),args.batch_size):
            start = i
            end = min(start+args.batch_size,len(data))
            Z,feats = pre_trained_model(data[start:end].to(device))
            probs.append(F.softmax(Z, dim=1).cpu().numpy())
            predictions.append(np.argmax(F.softmax(Z, dim=1).cpu().numpy(),axis=1))
            feats = feats.cpu().numpy()
            features.append(feats)
        probs = np.concatenate(probs)
        features = np.concatenate(features)
        predictions = np.concatenate(predictions)
    if get_probs:
        return features, probs
    return features,predictions
