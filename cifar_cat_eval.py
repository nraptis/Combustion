from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from cifar_control_nnet import CifarControlNet
from cifar_loader import pick_device

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = "./data"
CHECKPOINT = "./checkpoints/cifar_control.pt"
CLASS_NAME = "cat"
NUM_CATS = 20


@torch.no_grad()
def main():
    device = pick_device("auto")
    print("Device:", device)

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    cfg = ckpt["train_config"]

    model = CifarControlNet(
        conv1_out=cfg["conv1_out"],
        conv2_out=cfg["conv2_out"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load CIFAR-10 training set (no augmentation!)
    tfm = transforms.ToTensor()
    ds = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=tfm,
    )

    class_to_idx = {name: i for i, name in enumerate(ds.classes)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    cat_label = class_to_idx[CLASS_NAME]

    # Collect first N cats
    images = []
    for img, label in ds:
        if label == cat_label:
            images.append(img)
            if len(images) == NUM_CATS:
                break

    if len(images) < NUM_CATS:
        raise RuntimeError("Not enough cat images found")

    correct = 0

    print(f"\nEvaluating first {NUM_CATS} '{CLASS_NAME}' images:\n")

    for i, img in enumerate(images):
        x = img.unsqueeze(0).to(device)  # (1,3,32,32)
        logits = model(x)
        probs = F.softmax(logits, dim=1)

        pred_idx = int(probs.argmax(dim=1))
        pred_name = idx_to_class[pred_idx]
        conf = float(probs[0, pred_idx])

        is_correct = (pred_idx == cat_label)
        correct += int(is_correct)

        status = "✓" if is_correct else "✗"

        print(
            f"[{i:02d}] pred={pred_name:<10} "
            f"conf={conf:0.3f} "
            f"{status}"
        )

    acc = correct / NUM_CATS
    print("\n----------------------------------")
    print(f"Cat accuracy: {correct}/{NUM_CATS} = {acc*100:.1f}%")
    print("----------------------------------")


if __name__ == "__main__":
    main()
