"""
Training and testing code for the model.
"""

import wandb
from tqdm import tqdm

from config import *

def train(model, loader, optimizer):
    """
    Train the given model with data from the given dataloader.
    """
    for _ in tqdm(range(NUM_EPOCHS)):
        train_epoch(model, loader, optimizer)

def train_epoch(model, loader, optimizer):
    """
    Train the model for a single epoch.
    """
    for batch in tqdm(loader):
        inp, label = batch
        # inp: (batch_size, 1, 28, 28)
        # The single channel 28x28 image of the digit
        # label.shape: (batch_size)
        # The label for the digit, as a raw integer

        # Zero out gradients that may be present from previous batches
        optimizer.zero_grad()

        # Get the model's predictions for the batch, in probabilitstic terms
        pred = model(inp)

        # Compute loss and gradients
        loss = model.loss(pred, label)
        loss.backward()

        # Backpropagate
        optimizer.step()

        # Gather data and report
        wandb.log({'loss': loss})
