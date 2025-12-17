from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CifarLoaders:
    train: DataLoader
    test: DataLoader


def pick_device(device_str: str = "auto") -> str:
    """
    device_str:
      - "auto" (default): choose mps/cuda/cpu
      - "mps" | "cuda" | "cpu": force a specific device
    """
    if device_str != "auto":
        return device_str

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    device: str = "auto",
) -> CifarLoaders:
    """
    Returns train/test CIFAR-10 DataLoaders with standard light augmentation on train only.
    Images are tensors in [0..1], shape (C,H,W) = (3,32,32).
    Labels are ints in [0..9].
    """
    resolved_device = pick_device(device)

    tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm_train)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm_test)

    pin = (resolved_device != "cpu")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return CifarLoaders(train=train_dl, test=test_dl)
