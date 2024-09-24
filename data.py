import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt

# Default values (Refer to the cell 10 of the data notebook to know how the values were obtained)
MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2470, 0.2435, 0.2616])

# ImageNet's mean and std values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]  

class BasicProcessing:
    def __init__(self, mean=MEAN, std=STD):
        """
        Initializes the BasicProcessing class with normalization parameters.

        Args:
            - mean (tuple): Mean values for normalization, default value MEAN.
            - std (tuple): Standard deviation values for normalization, default value STD.

        Methods:
            - __call__(image): Applies the processing transformations to the input image, returning the processed image.
        """
        
        self.processing = v2.Compose([
            v2.Resize((32, 32)),
            v2.ToImage(),  
            v2.ToDtype(torch.float32, scale=True),    
            v2.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, img):
        return self.processing(img)
                        
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getClassNames(file='./data/cifar-10-batches-py/batches.meta'):
    """
    Loads the class names from a CIFAR-10 metadata file.

    Args:
        - file (str, optional): The path to the metadata file that contains the CIFAR-10 class names.
                               Defaults to './data/cifar-10-batches-py/batches.meta'.

    Returns:
        - List[str]: A list of class names decoded from bytes to UTF-8 strings.
    """
    
    dict_names = unpickle(file)
    return [names.decode('utf-8') for names in dict_names[b'label_names']]

def getDatasets(data_path='./data/cifar-10-batches-py/', transform=BasicProcessing()):
    """
    Loads the CIFAR-10 training and validation datasets with optional processing.

    Args:
        - data_path (str, optional): The path to the directory containing CIFAR-10 data.
                                     Defaults to './data/cifar-10-batches-py/'.
        - transform (callable, optional): A callable processing object to apply transformations to the data.
                                          Defaults to BasicProcessing().

    Returns:
        - tuple: A tuple containing the training dataset and the validation dataset:
                 (cifar10_train, cifar10_val).
    """
    
    # Load CIFAR10 datasets with or without processing
    cifar10_train = datasets.CIFAR10(data_path, train=True, transform=transform, download=True)
    cifar10_val = datasets.CIFAR10(data_path, train=False, transform=transform, download=True)

    return cifar10_train, cifar10_val


def getDataLoaders(batch_size=64, shuffle=True, transform=BasicProcessing()):
    """
    Creates and returns data loaders for CIFAR-10 training and validation datasets.

    Args:
        - batch_size (int, optional): The number of samples per batch to load. Defaults to 64.
        - shuffle (bool, optional): Whether to shuffle the dataset between epochs. Defaults to True.
        - transform (callable, optional): A callable object to apply transformations to the datasets. Defaults to BasicProcessing().

    Returns:
        - tuple: A tuple containing the training and validation data loaders:
                 (train_dataloader, val_dataloader).
    """
    cifar10_train, cifar10_val = getDatasets(transform=transform)
    
    train_dataloader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)
    val_dataloader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)
    
    return train_dataloader, val_dataloader

# Processing found on the Internet 
internet_processing = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(20),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
    v2.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False),
])
        