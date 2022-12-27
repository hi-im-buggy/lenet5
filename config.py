"""
Hyperparameters for training the model.
"""
import torch
import wandb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = './models/'
MODEL_LOAD_PATH = './models/model-2022-12-28-01:45:50-0'

training = True

wandb.config = {
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
}

