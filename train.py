"""
Training and testing code for the model.
"""

import wandb
from tqdm import tqdm

from config import *

def train(model, train_loader, val_loader, optimizer):
    """
    Train the given model with data from the given dataloader.
    """
    for i in tqdm(range(NUM_EPOCHS), desc="Epoch"):
        model.train(True)
        train_epoch(model, train_loader, optimizer, i)

        # Calculate and report validation loss
        model.train(False)
        validate_epoch(model, val_loader, i)

def train_epoch(model, loader, optimizer, epoch_idx):
    """
    Train the model for a single epoch.
    """
    running_loss = 0
    last_loss = 0
    for i, batch in tqdm(enumerate(loader), desc="Batch"):
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
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss/100
            wandb.log({
                'epoch': epoch_idx + 1,
                'batch': i + 1,
                'train loss': last_loss,
                })

def validate_epoch(model, loader, epoch_idx):
    """
    Test the model against the validation set.
    """
    running_loss = 0
    for i, batch in tqdm(enumerate(loader), desc="Batch"):
        inp, label = batch
        # Run inference on the current minibatch
        pred = model(inp)
        loss = model.loss(pred, label)

        # Add to the running loss calculation
        running_loss += loss

    avg_loss = running_loss / (i + 1)
    wandb.log({
        'epoch': epoch_idx + 1,
        'val loss': avg_loss
        })

def test(model, loader):
    running_loss = 0

    for i, batch in tqdm(enumerate(loader), desc="Batch"):
        inp, label = batch
        # Run inference on the current minibatch
        pred = model(inp)
        loss = model.loss(pred, label)

        # Add to the running loss calculation
        running_loss += loss

    avg_loss = running_loss / (i + 1)
    wandb.log({
        'test loss': avg_loss
        })
