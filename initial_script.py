import os
path = input("Enter a path to hold the data and results (press enter for current path): ")
if path=='':
    path =os.getcwd()

os.makedirs(path+'/DeepDIG')
os.makedirs(path+'/DeepDIG/Data')
os.makedirs(path+'/DeepDIG/Data/CIFAR10')
os.makedirs(path+'/DeepDIG/Data/FASHIONMNIST')
os.makedirs(path+'/DeepDIG/Data/MNIST')
os.makedirs(path+'/DeepDIG/PreTrainedModels/CIFAR10')
os.makedirs(path+'/DeepDIG/PreTrainedModels/FASHIONMNIST')
os.makedirs(path+'/DeepDIG/PreTrainedModels/MNIST')
