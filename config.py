"""
Hyperparameters for training the model.
"""
import torch
import wandb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

wandb.config = {
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
}

