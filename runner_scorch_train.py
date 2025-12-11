# runner_scorch_train.py

from __future__ import annotations

import torch
import torch.nn as nn

from scorch_ext.scorch_dataset import ScorchPatchClassificationDataset


class SmallConvNet(nn.Module):
    """
    Tiny CNN that works for variable H,W:
    we use AdaptiveAvgPool2d(1,1) so the final feature size is always 16.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Global average pool → always (N, 16, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),             # (N,16,1,1) → (N,16)
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def runner_scorch_train():
    print("=== SCORCH TRAIN START ===")

    # 1) Build dataset (no resizing, no batching concerns)
    dataset = ScorchPatchClassificationDataset(
        annotations_subdir="training",
        images_subdir="training",
        grayscale=True,
    )

    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0]
    num_classes = len(dataset.class_to_index)

    print(f"[ScorchTrain] in_channels = {in_channels}, num_classes = {num_classes}")
    print(f"[ScorchTrain] classes = {dataset.index_to_class}")

    # 2) Model, loss, optimizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SmallConvNet(in_channels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3) Simple SGD with batch_size = 1 (no DataLoader)
    epochs = 512
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i in range(len(dataset)):
            x, y = dataset[i]         # x: (C,H,W), y: scalar class index

            x = x.unsqueeze(0).to(device)   # → (1,C,H,W)
            y = y.unsqueeze(0).to(device)   # → (1,)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += 1

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        print(f"[Epoch {epoch+1}/{epochs}] loss={avg_loss:.4f}, acc={acc:.3f}")

    # After training finishes
    save_path = "hello_torch_2.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[ScorchTrain] Model saved to {save_path}")

    print("=== SCORCH TRAIN DONE ===")


if __name__ == "__main__":
    runner_scorch_train()
