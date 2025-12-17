from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# -----------------------------
# Config knobs
# -----------------------------
DATA_ROOT = "./data"
CLASS_NAME = "cat"
NUM_PREVIEW = 20
REFERENCE_INDEX = 13          # which cat (0..NUM_PREVIEW-1) to save
OUTPUT_PNG = "./reference_cat.png"


def main():
    tfm = transforms.ToTensor()

    ds = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=tfm,
    )

    # CIFAR-10 class mapping
    class_to_idx = {name: i for i, name in enumerate(ds.classes)}
    target_label = class_to_idx[CLASS_NAME]

    # Collect first N cat images
    cat_images: List[torch.Tensor] = []

    for img, label in ds:
        if label == target_label:
            cat_images.append(img)
            if len(cat_images) == NUM_PREVIEW:
                break

    if len(cat_images) < NUM_PREVIEW:
        raise RuntimeError(f"Only found {len(cat_images)} '{CLASS_NAME}' images")

    # -----------------------------
    # Plot grid
    # -----------------------------
    cols = 5
    rows = (NUM_PREVIEW + cols - 1) // cols

    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i, img in enumerate(cat_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.permute(1, 2, 0))  # CHW -> HWC
        plt.title(f"{CLASS_NAME} #{i}")
        plt.axis("off")

    plt.suptitle(f"First {NUM_PREVIEW} CIFAR-10 '{CLASS_NAME}' images", fontsize=14)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Save reference image
    # -----------------------------
    ref = cat_images[REFERENCE_INDEX]
    os.makedirs(os.path.dirname(OUTPUT_PNG) or ".", exist_ok=True)

    plt.imsave(
        OUTPUT_PNG,
        ref.permute(1, 2, 0).numpy()
    )

    print(f"Saved reference image: {OUTPUT_PNG}")
    print(f"Class: {CLASS_NAME}, index in preview set: {REFERENCE_INDEX}")


if __name__ == "__main__":
    main()
