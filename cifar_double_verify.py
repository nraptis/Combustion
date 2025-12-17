from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from cifar_control_nnet import CifarControlNet
from cifar_loader import pick_device

import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
CHECKPOINT = "./checkpoints/cifar_control.pt"
REFERENCE_PNG = "./reference_cat.png"
TOPK = 5

# If True: do a "video-style" nearest upscale to 128 then downscale back to 32,
# to see if that pipeline changes the prediction.
DO_UP_UP_DOWN = True
UP_SIZE = 128

def load_png_as_cifar_tensor(path: str) -> torch.Tensor:
    """
    Returns x shaped (1,3,32,32), float32 in [0..1].
    """
    img = Image.open(path).convert("RGB")
    if img.size != (32, 32):
        raise ValueError(f"{path} must be 32x32, got {img.size}")

    x = T.ToTensor()(img).unsqueeze(0)  # (1,3,32,32)
    return x


def pretty_topk(probs: torch.Tensor, classes: list[str], k: int):
    vals, idxs = torch.topk(probs, k=k, dim=1)
    vals = vals[0].tolist()
    idxs = idxs[0].tolist()
    for rank, (p, i) in enumerate(zip(vals, idxs), start=1):
        print(f"  {rank:>2}. {classes[i]:<10}  p={p:.4f}")


@torch.no_grad()
def run_once(model, x, classes: list[str], tag: str):
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred_idx = int(probs.argmax(dim=1))
    conf = float(probs[0, pred_idx])

    print(f"\n[{tag}] pred={classes[pred_idx]}  conf={conf:.4f}")
    print(f"[{tag}] top-{TOPK}:")
    pretty_topk(probs, classes, TOPK)


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

    # CIFAR-10 class names in standard order
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

    # Load reference image
    x = load_png_as_cifar_tensor(REFERENCE_PNG).to(device)

    # Pass 1: normal
    run_once(model, x, classes, tag="pass1")

    # Pass 2: identical second run (useful to confirm determinism)
    run_once(model, x, classes, tag="pass2")


    up = F.interpolate(x, size=(UP_SIZE, UP_SIZE), mode="nearest")
    up_img = up.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(up_img)
    plt.title("Upscaled (nearest)")
    plt.axis("off")
    plt.imsave(
    "reference_cat_upscaled.png",
    up.squeeze(0).permute(1, 2, 0).cpu().numpy()
)
    plt.show()


if __name__ == "__main__":
    main()
