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

data_dir = args.project_dir+'/Data/MNIST/'
transform = transforms.ToTensor()

def get_original_data(split='train'):
    if split =='test':
        data_name_file = 'test.pt'
    elif split =='train':
        data_name_file = 'training.pt'
    _data, labels = torch.load(data_dir+ data_name_file)
    data = []
    for img in _data:
       data.append(transform(Image.fromarray(img.numpy(), mode='L')))
    data= torch.stack(data)
    return data, labels

def get_class_specific_data(Class, split='train'):

    if split =='test':
        data_name_file = 'test.pt'
    elif split =='train':
        data_name_file = 'training.pt'
    _data, _labels = torch.load(data_dir+ data_name_file)

    data, labels =[],[]


    for i, (label, sample) in enumerate(zip(_labels,_data)):
        if label == Class:
            data.append(transform(Image.fromarray(sample.numpy(), mode='L')))
            labels.append(label)
    data = torch.stack(data)
    labels = torch.stack(labels)
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


