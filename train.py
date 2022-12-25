"""
Training and testing code for the model.
"""

import wandb

def train(model, loader):
    """
    Train the given model with data from the given dataloader.
    """
    pass

def train_epoch(model, loader):
    """
    Train the model for a single epoch.
    """
    for i, batch in enumerate(loader):
        inp, label = batch
        # inp: (batch_size, 1, 28, 28)
        # The single channel 28x28 image of the digit
        # label.shape: (batch_size)
        # The label for the digit, as a raw integer

        batch_size = inp.shape[0]

        # Zero out gradients that may be present from previous batches
        model.optim.zero_grad()

        # Get the model's predictions for the batch, in probabilitstic terms
        pred = model(inp)

        # Compute loss and gradients
        loss = model.loss(pred, label)
        loss.backward()

        # Backpropagate
        model.optim.step()

        # Gather data and report
        wandb.log({'loss': loss})
