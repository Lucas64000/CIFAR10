# PyTorch core libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set random seed
torch.manual_seed(123)

# Pre-trained models and transforms from torchvision
from torchvision import models
from torchvision.transforms import v2

# Data utilities for loaders and processing
from data import getDataLoaders, getClassNames
from data import BasicProcessing, internet_processing

# Utility libraries
from collections import OrderedDict
import datetime
import os
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# TensorBoard for logging
from torch.utils.tensorboard import SummaryWriter

# Confusion matrix tools from sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Basic ResNet model
class ResNet(nn.Module):
    """
    A ResNet18 model tailored for CIFAR-10 dataset by modifying the input layer and the fully connected output layer.

    Args:
        - weights (optional): Pretrained weights to load for the ResNet18 model, defaults to None.

    Attributes:
        - model (nn.Module): The ResNet18 architecture with customized layers for CIFAR-10.
    """
    
    def __init__(self, weights=None):
        super().__init__()

        self.model = models.resnet18(weights=weights)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        return self.model(x)

def model_accuracy(model, train_loader, val_loader, device):
    """
    Calculates and prints the model's accuracy and class distribution for both training and validation sets.

    Args:
        - model (torch.nn.Module): The model to evaluate.
        - train_loader (torch.utils.data.DataLoader): Training data.
        - val_loader (torch.utils.data.DataLoader): Validation data.
        - device (str): Device to run the model ('cpu' or 'cuda').
    """
    
    model = model.to(device=device)
    model.eval()
    for name, loader in zip((['train', 'val']), ([train_loader, val_loader])):
        correct = 0
        count = torch.zeros(10).to(device)
        total = len(loader.dataset)
        for imgs, labels in loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
           
            with torch.inference_mode():
                outputs = model(imgs)
                pred = torch.argmax(outputs, dim=1)
            correct += int((pred == labels).sum())
            count += torch.bincount(pred, minlength=10)

        print(f"Score {name}: {correct} / {total}",
              f"\nAccuracy {name}: {(correct / total)*100:.2f}%",
              f"\nDistribution {name} (in %): [{', '.join([f'{(c / total * 100):.2f}' for c in count])}]")

        print()

def training_loop(n_epochs, loader, model, optimizer, loss_fn, device):
    """
    Trains the model and prints the loss and accuracy for each epoch.

    Args:
        - n_epochs (int): Number of epochs to train the model.
        - loader (torch.utils.data.DataLoader): Training data.
        - model (torch.nn.Module): The model to train.
        - optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        - loss_fn (torch.nn.Loss): Loss function.
        - device (str): Device to run the model ('cpu' or 'cuda').
    """
    
    model = model.to(device)

    # Loss is computed over batches, accuracy is computed over individual items
    total_items = len(loader.dataset)
    total_batches = len(loader)
    
    for epoch in range(1, n_epochs + 1):
        model.train()  
        loss_train = 0.0
        correct = 0
        total = 0 

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()

        average_loss = loss_train / total_batches
        accuracy = (correct / total_items) * 100  
        
        print(f"Epoch [{epoch}/{n_epochs}]: Loss_train = {average_loss:.4f}, Accuracy = {accuracy:.2f}%")

class BasicBlock(nn.Module):
    """
    A basic residual block for building deep networks.

    Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - downsample (bool, optional): If True, apply downsampling to reduce the spatial size. Defaults to False.

    Attributes:
        - conv1 (nn.Conv2d): First convolution layer.
        - bn1 (nn.BatchNorm2d): First batch normalization layer.
        - relu (nn.ReLU): ReLU activation function.
        - conv2 (nn.Conv2d): Second convolution layer.
        - bn2 (nn.BatchNorm2d): Second batch normalization layer.
        - downsample (nn.Sequential, optional): Downsampling layer, if applicable.
    """

    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if downsample:
            self.downsample = nn.Sequential(OrderedDict([
                                  ('conv3', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)),
                                  ('bn3', nn.BatchNorm2d(num_features=out_channels)),
                                ]))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        #  If downsampling is applied, we need to adjust the residual's dimensions
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual

        return self.relu(out)


class ResNetLayer(nn.Module):
    """
    A custom ResNet layer block for building a deep convolutional neural network.

    Args:
        - layers (int): Number of layers to include in the block.
        - out_channels (int, optional): Number of output channels for the first convolution layer. Defaults to 64.
        - dropout (float, optional): Dropout probability. Defaults to 0.

    Attributes:
        - in_channels (int): Number of input channels.
        - conv1 (nn.Conv2d): First convolution layer with 3 input channels.
        - bn1 (nn.BatchNorm2d): First batch normalization layer.
        - relu (nn.ReLU): ReLU activation function.
        - layers (nn.ModuleDict): Dictionary of residual layers.
        - avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        - fc (nn.Linear): Fully connected layer for classification.
        - dropout (nn.Dropout, optional): Dropout layer applied after the fully connected layer, if specified.

    Methods:
        - make_layer(downsample): Creates a residual layer with or without downsampling.
        - predict(x): Runs inference, returning predicted class and probabilities.
    """

    def __init__(self, layers, out_channels=64, dropout=0):
        super().__init__()
        self.in_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleDict()

        # Creation of the layers 
        for num_layer in range(1, layers+1):
            # No downsampling in the first layer
            downsample = True if num_layer > 1 else False

            layer_name = f'layer{num_layer}'
            layer = self.make_layer(downsample)
            self.layers[layer_name] = layer

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=self.in_channels, out_features=10)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        
    def make_layer(self, downsample):
        if downsample:
            layer = nn.Sequential(OrderedDict([
                ('block1', BasicBlock(in_channels=self.in_channels, out_channels=self.in_channels*2, downsample=True)),
                ('block2', BasicBlock(in_channels=self.in_channels*2, out_channels=self.in_channels*2))
            ]))
            # Update in_channels after downsampling
            self.in_channels *= 2
        else:
            layer = nn.Sequential(OrderedDict([
                ('block1', BasicBlock(in_channels=self.in_channels, out_channels=self.in_channels)),
                ('block2', BasicBlock(in_channels=self.in_channels, out_channels=self.in_channels))
            ]))

        return layer

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        for _, layer in self.layers.items():
            out = layer(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        
        if self.dropout:
            out = self.dropout(out)
        
        return out

    def predict(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_names = getClassNames()

        with torch.inference_mode():
            self.eval()
            self = self.to(device)
            x = x.to(device)
            
            logits = self(x)
            pred = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)

            probs_np = probs.cpu().numpy()
            dict_probs = {class_names[i]: f"{prob*100:.2f}" for i, prob in enumerate(probs_np[0])} # probs_np.shape = [1, 10]

            class_pred = class_names[pred.item()]
        return class_pred, dict_probs


def train_model(n_epochs, model, train_loader, val_loader, loss_fn, optimizer, device, log_dir="./runs"):
    """
    Trains the model and logs metrics such as loss and accuracy for both training and validation sets using TensorBoard.

    Args:
        - n_epochs (int): Number of epochs for training.
        - model (torch.nn.Module): Model to train.
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        - loss_fn (torch.nn.Loss): Loss function to compute the loss.
        - optimizer (torch.optim.Optimizer): Optimizer for model parameter updates.
        - device (str): Device to run the training ('cpu' or 'cuda').
        - log_dir (str, optional): Directory to store TensorBoard logs. Defaults to './runs'.

    Details:
        - This is an enhanced version of the previous training function that logs the training and validation metrics 
          (loss and accuracy) to TensorBoard for better visualization and tracking.
        - For each epoch, the function logs:
            - The average training loss.
            - The training accuracy.
            - The validation loss.
            - The validation accuracy.
    """

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    total_train_batches = len(train_loader)
    total_train_items = len((train_loader.dataset)

    total_val_batches = len(val_loader)                    
    total_val_items = len(val_loader.dataset)
    
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        # Loss/train metric
        loss_train = 0.0
        # Acc/train metric
        correct_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            logits = model(imgs)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            predicted = torch.argmax(logits, dim=1)
            correct_train += (predicted == labels).sum().item()
            
        total_loss_train = loss_train / total_train_batches
        total_acc_train = (correct_train / total_train_items) * 100 

        model.eval()
        # Loss/val metric
        loss_val = 0.0
        # Loss/accuracy metric
        correct_val = 0

        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                
                logits = model(imgs)
                loss = loss_fn(logits, labels)

                loss_val += loss.item()
                predicted = torch.argmax(logits, dim=1)
                correct_val += (predicted == labels).sum().item()

        total_loss_val = loss_val / total_val_batches
        total_acc_val = (correct_val / total_val_items) * 100

        # Metrics logs
        writer.add_scalar('Loss/train', total_loss_train, epoch)
        writer.add_scalar('Accuracy/train', total_acc_train, epoch)
        writer.add_scalar('Loss/validation', total_loss_val, epoch)
        writer.add_scalar('Accuracy/validation', total_acc_val, epoch)

        print(f'Epoch [{epoch+1}/{n_epochs}]')
        print(f'Train Loss: {total_loss_train:.4f}, Train Accuracy: {total_acc_train:.2f}%')
        print(f'Validation Loss: {total_loss_val:.4f}, Validation Accuracy: {total_acc_val:.2f}%')

    writer.close()

def display_matrix(y_true, y_pred):
    """
    Displays the confusion matrix for true and predicted labels.

    Args:
        - y_true (list[int]): True labels.
        - y_pred (list[int]): Predicted labels.
    """

    class_names = getClassNames()
    
    y_true = [class_names[label] for label in y_true]
    y_pred = [class_names[pred] for pred in y_pred]
    
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(include_values=True, cmap='inferno', ax=None, xticks_rotation=45)
    
    fig = disp.ax_.get_figure() 
    fig.set_figwidth(7)
    fig.set_figheight(7) 

def display_matrices(model_names, y_true1, y_pred1, y_true2, y_pred2):
    """
    Displays confusion matrices for two sets of true and predicted labels.

    Args:
        - model_names (list[str]): Names of the models.
        - y_true1 (list[int]): True labels for the first model.
        - y_pred1 (list[int]): Predicted labels for the first model.
        - y_true2 (list[int]): True labels for the second model.
        - y_pred2 (list[int]): Predicted labels for the second model.
    """

    class_names = getClassNames()

    y_true1 = [class_names[label] for label in y_true1]
    y_pred1 = [class_names[class_pred] for class_pred in y_pred1]
    
    y_true2 = [class_names[label] for label in y_true2]
    y_pred2 = [class_names[class_pred] for class_pred in y_pred2]

    cm1 = confusion_matrix(y_true1, y_pred1, labels=class_names)
    cm2 = confusion_matrix(y_true2, y_pred2, labels=class_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=class_names)
    disp1.plot(include_values=True, cmap='inferno', ax=axes[0], xticks_rotation=45)
    axes[0].set_title(model_names[0])

    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=class_names)
    disp2.plot(include_values=True, cmap='inferno', ax=axes[1], xticks_rotation=45)
    axes[1].set_title(model_names[1])
    
    fig.set_figwidth(14)  
    fig.set_figheight(7) 
    
    plt.tight_layout()
    plt.show()

def get_predictions(model, loader, device):
    """
    Gets predictions from the model for a given data loader.

    Args:
        - model (torch.nn.Module): The model to use for predictions.
        - loader (torch.utils.data.DataLoader): DataLoader for the input data.
        - device (str): Device to run the model ('cpu' or 'cuda').

    Returns:
        - list[int]: True labels.
        - list[int]: Predicted labels.
    """

    model = model.to(device)
    model.eval()  
    all_labels = []
    all_preds = []
    
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy()) # Tensors must be on the cpu to be converted into numpy arrays
            all_preds.extend(preds.cpu().numpy()) 
    
    return all_labels, all_preds

def test_image(image_path, model, model_path):
    """
    Tests an image with a trained model and displays the result.

    Args:
        - image_path (str): Path to the image file.
        - model (nn.Module): The trained model to use for prediction.
        - model_path (str): Path to the model weights.

    Returns:
        - tuple: (predicted_class, probabilities_dict)
    """
    img = Image.open(image_path)
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    img = basic_processing(img)
    img = img.unsqueeze(0)  # Adds batch dimension

    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model.predict(img)
