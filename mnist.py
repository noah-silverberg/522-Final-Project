# mlp_mnist.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPMNIST(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # one dropout module reused for both hidden layers
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # MNIST is (N, 1, 28, 28)
        x = self.flatten(x)                 # (N, 784)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)                 # dropout on hidden 1 activations

        x = F.relu(self.fc2(x))
        x = self.dropout(x)                 # dropout on hidden 2 activations

        logits = self.fc3(x)                # no dropout here
        return logits
