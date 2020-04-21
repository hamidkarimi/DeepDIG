from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(50, 10))

    def forward(self, x):
        x = x.view(-1, 784)
        features = self.net(x)
        x = self.fc(features)
        return x, features
