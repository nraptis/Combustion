# penguin_train_torch.py
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from penguin_loader import load_penguin_dataset, PenguinDataset


class PenguinNet(nn.Module):
    """
    Simple MLP for tabular penguin features -> species logits.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_dataloaders(
    ds: PenguinDataset,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap numpy arrays in TensorDatasets + DataLoaders.
    """
    x_train = torch.from_numpy(ds.x_train)
    y_train = torch.from_numpy(ds.y_train)

    x_val = torch.from_numpy(ds.x_val)
    y_val = torch.from_numpy(ds.y_val)

    x_test = torch.from_numpy(ds.x_test)
    y_test = torch.from_numpy(ds.y_test)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Return (avg_loss, accuracy) over a DataLoader.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += float(loss.item()) * x.size(0)

            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1) if total > 0 else 0.0
    return avg_loss, acc


def train_penguin_net(
    csv_path: str = "penguins_cleaned.csv",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    random_seed: int = 1337,
):
    # ------------------------------------------------------
    # Repro-ish seeds
    # ------------------------------------------------------
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[penguin_train_torch] Using device: {device}")

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    ds = load_penguin_dataset(
        csv_path=csv_path,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    num_classes = len(ds.species_to_index)
    in_features = ds.x_train.shape[1]

    print(f"[penguin_train_torch] Classes: {ds.species_to_index}")
    print(f"[penguin_train_torch] Input dim: {in_features}")
    print(f"[penguin_train_torch] Train/Val/Test sizes: "
          f"{ds.x_train.shape[0]} / {ds.x_val.shape[0]} / {ds.x_test.shape[0]}")

    train_loader, val_loader, test_loader = make_dataloaders(
        ds, batch_size=batch_size
    )

    # ------------------------------------------------------
    # Model / optimizer / loss
    # ------------------------------------------------------
    model = PenguinNet(in_features=in_features, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------
    # Training loop
    # ------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            preds = logits.argmax(dim=1)
            running_correct += int((preds == y).sum().item())
            running_total += x.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1) if running_total > 0 else 0.0

        val_loss, val_acc = evaluate_model(model, val_loader, device=device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

    # ------------------------------------------------------
    # Final test eval
    # ------------------------------------------------------
    test_loss, test_acc = evaluate_model(model, test_loader, device=device)
    print(
        f"[penguin_train_torch] TEST: "
        f"loss={test_loss:.4f}, acc={test_acc:.3f}"
    )

    # Optionally return the trained model + dataset
    return model, ds


if __name__ == "__main__":
    # Assumes penguins_cleaned.csv is in the same directory
    train_penguin_net()
