import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarNet(nn.Module):
    def __init__(self, conv1_out=24, conv2_out=48, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(3, conv1_out, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size, padding=pad)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)

        # CIFAR is 32x32:
        # after pool twice: 32->16->8
        self.fc = nn.Linear(conv2_out * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B, conv1_out, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))   # (B, conv2_out, 8, 8)
        x = self.drop(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
