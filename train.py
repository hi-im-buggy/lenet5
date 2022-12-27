"""
Training and testing code for the model.
"""

import torch
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from config import *

best_loss = 1e9

def train(model, train_loader, val_loader, optimizer):
    """
    Train the given model with data from the given dataloader.
    """
    for i in tqdm(range(NUM_EPOCHS), desc="Train epoch"):
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
    for i, batch in enumerate(tqdm(loader, desc="Train batch")):
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
    global best_loss
    running_loss = 0
    for i, batch in enumerate(tqdm(loader, desc="Validation batch")):
        inp, label = batch
        # Run inference on the current minibatch
        pred = model(inp)
        loss = model.loss(pred, label)

        # Add to the running loss calculation
        running_loss += loss

    avg_loss = running_loss.item() / len(loader)
    wandb.log({
        'epoch': epoch_idx + 1,
        'val loss': avg_loss
        })

    # Save the model, if it has better loss than the best so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        model_path = MODEL_SAVE_PATH + 'model-{}-{}'.format(timestamp, epoch_idx)
        torch.save(model.state_dict(), model_path)

def test(model, loader):
    """
    Calculate the testing accuracy of model.
    """
    model.train(False)

    hits = 0
    samples = 0

    for i, batch in enumerate(tqdm(loader, desc="Test batch")):
        inp, label = batch
        pred = model(inp)
        top = torch.argmax(pred, dim=-1)

        # Count the hits and total samples for each batch and add them to the totals
        batch_hits = torch.count_nonzero(top == label).item()
        batch_size = inp.shape[0]

        hits += batch_hits
        samples += batch_size

        # Plot the first image in the batch and log its prediction and true label
        img = inp[0].permute(1, 2, 0)
        plt.imshow(img.numpy())

        wandb.log({
            'image': img,
            'pred': top[0],
            'target': label[0],
            })

    accuracy = hits / samples

    print(f'Accuracy: {accuracy}')
    wandb.log({'Test accuracy': accuracy})
