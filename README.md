# LeNet5
This is an implementation of LeCun et al.'s classic CNN architecture, trained
on the MNIST dataset for handwritten digit recognition.

## Architecture overview
```
Input: 32x32 (or 28x28 with appropriate padding for MNIST)
C1 (28x28): convolution layer, 5x5 kernel, stride of 1, no padding, 6 channels
S2 (14x14): avg pooling layer, 2x2 kernel, stride of 2, no padding, 6 channels
C3 (10x10): convolution layer, 5x5 kernel, stride of 1, no padding, 16 channels
S4 (5x5): avg pooling layer, 5x5 kernel, stride of 2, no padding, 16 channels
C5 (5x5): convolution layer, 5x5, stride of 1, no padding, 120 channels
F6: fully connected layer, 120 to 84
F7: fully connected layer, 84 to 10
```
