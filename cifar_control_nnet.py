# cifar_control_nnet.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarControlNet(nn.Module):
    """
    A simple CIFAR-10 CNN baseline.
    First conv uses 24 kernels (channels) as requested.
    """

    def __init__(
        self,
        conv1_out: int = 24,
        conv2_out: int = 48,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(3, conv1_out, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size, padding=pad)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)

        # CIFAR: 32x32 -> pool -> 16x16 -> pool -> 8x8
        self.fc = nn.Linear(conv2_out * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 24, 16, 16) default
        x = self.pool(F.relu(self.conv2(x)))   # (B, 48, 8, 8) default
        x = self.drop(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
