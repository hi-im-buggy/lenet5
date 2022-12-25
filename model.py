"""
Definition of the model architecture.
"""
from torch import nn, optim

from config import *

class LeNet5(nn.Module):
    """
    LeNet5 model architecture
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        # Model training params
        self.optim = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.CrossEntropyLoss()

        # Convolution layers
        # Padding of 2 on the first convolutional layer allows for 28x28 input from  MNIST
        # to be treated like 32x32 input.
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)

        # Pooling layer
        self.pool = nn.AvgPool2d(5, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

        # Activation layers
        self.relu = nn.ReLU()

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Takes a batch of 28x28 images.
        Returns the probability distribution for recognizing the digit.
        """
        # x.shape: (N, 28, 28) or (N, 1, 28, 28), assuming former
        x.unsqueeze(1)

        # x.shape: (N, 1, 28, 28)
        x = self.relu(self.conv1(x))
        # x.shape: (N, 6, 28, 28)
        x = self.pool(x)

        # x.shape: (N, 6, 14, 14)
        x = self.relu(self.conv2(x))
        # x.shape: (N, 16, 10, 10)
        x = self.pool(x)

        # x.shape: (N, 16, 5, 5)
        x = self.relu(self.conv3(x))

        # x.shape: (N, 120, 1, 1)
        x.squeeze(-1).squeeze(-1)

        # x.shape: (N, 120)
        x = self.relu(self.fc1(x))
        # x.shape: (N, 84)
        x = self.relu(self.fc1(x))

        # x.shape: (N, 10)
        prob = self.softmax(x)
        return prob
