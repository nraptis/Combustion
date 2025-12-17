from __future__ import annotations

import torch
import torch.nn.functional as F

from cifar_loader import get_cifar10_loaders, pick_device
from cifar_control_nnet import CifarControlNet

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

    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    ckpt_path = "./checkpoints/cifar_control.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["train_config"]

    device = pick_device("auto")
    loaders = get_cifar10_loaders(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        device=device,
    )

    model = CifarControlNet(
        conv1_out=cfg["conv1_out"],
        conv2_out=cfg["conv2_out"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    loss, acc = evaluate(model, loaders.test, device)
    print("Checkpoint:", ckpt_path)
    print(f"Test loss: {loss:.4f}")
    print(f"Test acc : {acc*100:.2f}%")


if __name__ == "__main__":
    main()
