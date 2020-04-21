from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self,dropout=0.0):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 784)
        self.dropout = dropout
    def encode(self, x):
        h1 = F.dropout(F.relu(self.fc1(x)),p=self.dropout)
        return h1

    def decode(self, z):
        h3 = F.dropout(F.relu(self.fc2(z)),p=self.dropout)
        return torch.sigmoid(self.fc3(h3))

    def forward(self, x):
        encoded = self.encode(x.view(-1, 784))
        decoded = self.decode(encoded)
        return encoded,decoded.view(decoded.size(0),1,28,28)
