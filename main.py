"""
Main driver code for model training and inference.
"""
import torch
import wandb
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

wandb.init(project="LeNet5 MNIST")

from config import *
from train import *
from model import *

t = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=(0), std=(1))])
train_set = MNIST('./data', train=True, download=True, transform=t)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)

model = LeNet5()

train_epoch(model, train_loader)

wandb.watch(model)
