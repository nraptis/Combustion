# cifar_control_train.py

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from cifar_loader import get_cifar10_loaders, pick_device
from cifar_control_nnet import CifarControlNet

@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"

    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 2

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # model knobs
    conv1_out: int = 24
    conv2_out: int = 48
    kernel_size: int = 3
    dropout: float = 0.1

    log_every: int = 100
    save_path: str = "./checkpoints/cifar_control.pt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dl, device: str):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss_sum += float(loss) * xb.size(0)

        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum())
        total += xb.size(0)

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    cfg = TrainConfig()

    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    print("Device:", device)

    loaders = get_cifar10_loaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )

    model = CifarControlNet(
        conv1_out=cfg.conv1_out,
        conv2_out=cfg.conv2_out,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for step, (xb, yb) in enumerate(loaders.train, start=1):
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            running_loss += float(loss) * xb.size(0)
            seen += xb.size(0)

            if step % cfg.log_every == 0:
                train_loss = running_loss / max(seen, 1)
                test_loss, test_acc = evaluate(model, loaders.test, device)
                print(
                    f"epoch {epoch:02d} step {step:04d} | "
                    f"train_loss {train_loss:.4f} | "
                    f"test_loss {test_loss:.4f} | test_acc {test_acc*100:.2f}%"
                )

        # End-of-epoch report
        train_loss = running_loss / max(seen, 1)
        test_loss, test_acc = evaluate(model, loaders.test, device)
        print(
            f"[EPOCH END] epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | test_loss {test_loss:.4f} | test_acc {test_acc*100:.2f}%"
        )

    # Save
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "train_config": cfg.__dict__,
        },
        cfg.save_path,
    )
    print("Saved checkpoint:", cfg.save_path)


if __name__ == "__main__":
    main()
