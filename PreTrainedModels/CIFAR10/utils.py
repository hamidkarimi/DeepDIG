from __future__ import print_function
import torch
import numpy as np
from torchvision import  transforms
import torchvision
import os
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import importlib

from DeepDIGCode import config
args= config.args

data_dir = args.project_dir+'/Data/CIFAR10/'
transform = transforms.ToTensor()

def get_train_data(Class=None):
    train_list = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    _data, _labels = [], []
    for file in train_list:
        entry =pickle.load(open(data_dir+file,'rb'),encoding='latin1')
        _data.append(entry['data'])
        _labels.append(entry['labels'])
    _data = np.concatenate(_data)
    _labels = np.concatenate(_labels)
    if Class is not None and Class >=0 and Class <=9:
        indices = np.where(_labels==Class)[0]
        _data = _data[indices]
        _labels = _labels[indices]

    _data = _data.reshape((len(_data), 3, 32, 32))
    _data = _data.transpose((0, 2, 3, 1))
    data = []
    for img in _data:
       data.append(transform(Image.fromarray(img)))
    data= torch.stack(data)
    return data,torch.from_numpy(_labels)

def get_test_data(Class=None):
    train_list = ['test_batch']
    _data, _labels = [], []
    for file in train_list:
        entry =pickle.load(open(data_dir+file,'rb'),encoding='latin1')
        _data.append(entry['data'])
        _labels.append(entry['labels'])
    _data = np.concatenate(_data)
    _labels = np.concatenate(_labels)

    if Class is not None and Class >=0 and Class <=9:
        indices = np.where(_labels==Class)[0]
        _data = _data[indices]
        _labels = _labels[indices]

    _data = _data.reshape((len(_data), 3, 32, 32))
    _data = _data.transpose((0, 2, 3, 1))
    data = []
    for img in _data:
       data.append(transform(Image.fromarray(img)))
    data= torch.stack(data)
    return data,torch.from_numpy(_labels)

def get_original_data(split='train'):
    if split =='train':
        data, labels = get_train_data()
        return data, labels
    else:
        data, labels = get_test_data()
        return data, labels
def get_class_specific_data(Class, split='train'):
    if split =='train':
        data, labels = get_train_data(Class=Class)
        return data, labels
    else:
        data, labels = get_test_data(Class=Class)
        return data, labels

def imshow(img,fname,show=True,title=""):
    img = torchvision.utils.make_grid(img.data)
    npimg = img.detach().cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.title(title)

    if show:
        plt.show()
    else:
        plt.savefig(fname)
def save_samples(dir,samples,filename,show=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for j,i in enumerate(range(0,len(samples),50)):
        start = i
        end = min(start+50,len(samples))
        imshow(samples[start:end], dir + filename + '_{}.png'.format(j), show=show)
