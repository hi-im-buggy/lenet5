"""
Main driver code for model training and inference.
"""
import torch
import wandb
from torch import optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

wandb.init(project="LeNet5 MNIST")

from config import *
from train import *
from model import *

t = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=(0), std=(1))])
dataset = MNIST('./data', train=True, download=True, transform=t)
train_set, val_set = random_split(dataset, [50000, 10000])
test_set = MNIST('./data', train=False, download=True, transform=t)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

model = LeNet5()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train(model, train_loader, val_loader, optimizer)
model.train(False)
test(model, test_loader)

wandb.watch(model)
