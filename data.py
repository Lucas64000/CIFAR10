import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from pathlib import Path

import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Creation of the list containing all class names 

def getClassNames(file='./data/cifar-10-batches-py/batches.meta'):
    dict_names = unpickle(file)
    return [names.decode('utf-8') for names in dict_names[b'label_names']] 

# Convert images to tensors and normalize their values using the computed mean and std (cf Notebook)

def getDatasets(data_path='./data/cifar-10-batches-py/', data_process=True):
    # If data_process is True, apply the transformations, otherwise set to None
    if data_process:
        # Mean and standard deviation for normalization (Refer to the data notebook, cell 10, for how the values were obtained.)
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])
        
        processing = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
    else:
        processing = None

    # Load CIFAR10 datasets with or without processing
    cifar10_train = datasets.CIFAR10(data_path, train=True, transform=processing, download=True)
    cifar10_val = datasets.CIFAR10(data_path, train=False, transform=processing, download=True)

    return cifar10_train, cifar10_val

def getDataLoaders(batch_size=64, shuffle=True):
    cifar10_train, cifar10_val = getDatasets()
    
    train_dataloader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, val_dataloader
        