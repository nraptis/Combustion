from __future__ import annotations

import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
INPUT_PNG = "./reference_cat.png"
OUTPUT_PNG = "./reference_cat_upscaled_128.png"
TARGET_SIZE = 128   # 32 -> 128
MODE = "nearest"    # this is the key


def main():
    # Load image as tensor (C,H,W) in [0,1]
    img = T.ToTensor()(plt.imread(INPUT_PNG))

    if img.shape[1] != 32 or img.shape[2] != 32:
        print("Warning: input image is not 32x32")

    # Add batch dim: (1,C,H,W)
    img_b = img.unsqueeze(0)

    # Dumb upscale
    up = torch.nn.functional.interpolate(
        img_b,
        size=(TARGET_SIZE, TARGET_SIZE),
        mode=MODE
    )

    # Back to (C,H,W)
    up = up.squeeze(0)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PNG) or ".", exist_ok=True)
    plt.imsave(
        OUTPUT_PNG,
        up.permute(1, 2, 0).numpy()
    )

    print("Saved:", OUTPUT_PNG)
    print("Input shape :", tuple(img.shape))
    print("Output shape:", tuple(up.shape))


if __name__ == "__main__":
    main()
