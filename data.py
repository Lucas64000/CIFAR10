import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

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

def getDatasets(data_path='./data/cifar-10-batches-py/', mean=None, std=None):
    # Default values (Refer to the cell 10 of the data notebook to know how the values were obtained)
    default_mean = torch.tensor([0.4914, 0.4822, 0.4465])
    default_std = torch.tensor([0.2470, 0.2435, 0.2616])
    
    # Use provided values or fall back to defaults
    mean = mean if mean is not None else default_mean
    std = std if std is not None else default_std

    # Define processing pipeline
    processing = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        v2.Normalize(mean=mean, std=std)
    ]) if mean is not None and std is not None else None

    # Load CIFAR10 datasets with or without processing
    cifar10_train = datasets.CIFAR10(data_path, train=True, transform=processing, download=True)
    cifar10_val = datasets.CIFAR10(data_path, train=False, transform=processing, download=True)

    return cifar10_train, cifar10_val

def getDataLoaders(batch_size=64, shuffle=True):
    cifar10_train, cifar10_val = getDatasets()
    
    train_dataloader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, val_dataloader
        