from __future__ import annotations

# cifar_modified_nnet.py

from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar_conv_2d_with_frames import CifarConv2dWithFrames


class CifarModifiedNet(nn.Module):
    """
    Same architecture as control, but conv1 is CifarConv2dWithFrames.
    Debug/video is OFF by default so eval is normal.
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

        self.conv1 = CifarConv2dWithFrames(3, conv1_out, kernel_size, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size, padding=pad)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(conv2_out * 8 * 8, num_classes)

    def set_video(
        self,
        enabled: bool,
        emit: Optional[Callable[[torch.Tensor, Dict], None]] = None,
        ref_index: int = 0,
        emit_mode: str = "snapshot",   # "snapshot" or "nudge"
        nudge_mode: str = "out_in",    # only used when emit_mode == "nudge"
    ):
        self.conv1.debug_enabled = enabled
        self.conv1.emit = emit
        self.conv1.ref_index = ref_index
        self.conv1.emit_mode = emit_mode
        self.conv1.nudge_mode = nudge_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
