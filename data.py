import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt

# Default values (Refer to the cell 10 of the data notebook to know how the values were obtained)
MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2470, 0.2435, 0.2616])

basic_processing = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD)
]) 

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

def getDatasets(data_path='./data/cifar-10-batches-py/', transform=basic_processing):
    # Load CIFAR10 datasets with or without processing
    cifar10_train = datasets.CIFAR10(data_path, train=True, transform=transform, download=True)
    cifar10_val = datasets.CIFAR10(data_path, train=False, transform=transform, download=True)

    return cifar10_train, cifar10_val

def getDataLoaders(batch_size=64, shuffle=True, transform=basic_processing):
    cifar10_train, cifar10_val = getDatasets(transform=transform)
    
    train_dataloader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)
    val_dataloader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)
    
    return train_dataloader, val_dataloader
        