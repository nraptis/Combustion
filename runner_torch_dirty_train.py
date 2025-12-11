# runner_torch_dirty_train.py

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from scorch_ext.data_loader import DataLoader as ScorchLoader
from scorch_ext.tensor_load_helpers import (
    iter_label_patches_from_pair_fixed_size_centered,
)
from scorch.scorch_tensor import ScorchTensor


# ================================================================
# Config
# ================================================================

FIXED_WIDTH = 92
FIXED_HEIGHT = 92

# multi-scale shrink range: 1.0 = original size, 0.5 = shrink to 50%
MIN_SCALE = 0.50
MAX_SCALE = 1.00  # don't enlarge beyond base canvas for now

EPOCHS = 160
LR = 1e-3
LAMBDA_MASK = 1.0  # weight for segmentation loss

CLASS_NAMES = ["Red", "Blue", "Green"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


# ================================================================
# Dataset: multi-scale, image + mask
# ================================================================

class ScorchMultiScaleSegDataset(torch.utils.data.Dataset):
    """
    Builds a list of base (image, mask, label_idx) patches using
    iter_label_patches_from_pair_fixed_size_centered, then on each
    __getitem__ applies a random multi-scale resampling to BOTH the
    image and the mask using PIL.

    Returns:
        x       : (1, H, W) float32 in [0,1]
        y_cls   : scalar long (class index)
        y_mask  : (1, H, W) float32 with values in {0,1}
    """

    def __init__(
        self,
        annotations_subdir: str = "training",
        images_subdir: str = "training",
        fixed_width: int = FIXED_WIDTH,
        fixed_height: int = FIXED_HEIGHT,
        grayscale: bool = True,
    ) -> None:
        super().__init__()
        self.fixed_width = int(fixed_width)
        self.fixed_height = int(fixed_height)
        self.grayscale = grayscale

        self.class_names = CLASS_NAMES
        self.class_to_index = CLASS_TO_INDEX

        self.samples: List[Tuple[ScorchTensor, ScorchTensor, int]] = []

        loader = ScorchLoader(
            annotations_subdir=annotations_subdir,
            images_subdir=images_subdir,
        )

        for pair_index, pair in enumerate(loader):
            for img_tensor, mask_tensor, label_name in (
                iter_label_patches_from_pair_fixed_size_centered(
                    pair=pair,
                    fixed_width=self.fixed_width,
                    fixed_height=self.fixed_height,
                    grayscale=self.grayscale,
                )
            ):
                if label_name not in self.class_to_index:
                    # You can tighten this if needed
                    continue

                label_idx = self.class_to_index[label_name]
                self.samples.append((img_tensor, mask_tensor, label_idx))

        print(
            f"[Dataset] Built {len(self.samples)} patches "
            f"from {annotations_subdir}/{images_subdir} with "
            f"fixed size = ({self.fixed_width}x{self.fixed_height})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------
    # Internal: apply random multi-scale resample to image+mask
    # ------------------------------------------------------------
    def _random_rescale_to_canvas(
        self,
        img_t: ScorchTensor,
        mask_t: ScorchTensor,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Given base ScorchTensors (1,H,W) for image and mask, convert to
        PIL images, rescale to a random smaller size, and center-paste
        into a fixed (fixed_width, fixed_height) canvas.

        Returns:
            (canvas_img, canvas_mask) as PIL "L" images.
        """
        # 1) ScorchTensor -> PIL
        img_pil = img_t.to_image()   # "L" or "RGB" depending on grayscale
        mask_pil = mask_t.to_image() # mask comes out as grayscale 0..255

        # We want both as single-channel "L" images
        if img_pil.mode != "L":
            img_pil = img_pil.convert("L")
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")

        base_w, base_h = img_pil.size

        # 2) Choose random scale factor in [MIN_SCALE, MAX_SCALE]
        s = random.uniform(MIN_SCALE, MAX_SCALE)

        new_w = max(1, int(round(base_w * s)))
        new_h = max(1, int(round(base_h * s)))

        # 3) Resize image and mask with SAME size
        img_resized = img_pil.resize(
            (new_w, new_h),
            resample=Image.BILINEAR,
        )
        mask_resized = mask_pil.resize(
            (new_w, new_h),
            resample=Image.NEAREST,  # keep mask crisp
        )

        # 4) New blank canvases
        canvas_img = Image.new("L", (self.fixed_width, self.fixed_height), 0)
        canvas_mask = Image.new("L", (self.fixed_width, self.fixed_height), 0)

        # Center offsets
        off_x = (self.fixed_width - new_w) // 2
        off_y = (self.fixed_height - new_h) // 2

        canvas_img.paste(img_resized, (off_x, off_y))
        canvas_mask.paste(mask_resized, (off_x, off_y))

        return canvas_img, canvas_mask

    def __getitem__(self, idx: int):
        img_t, mask_t, label_idx = self.samples[idx]

        canvas_img, canvas_mask = self._random_rescale_to_canvas(img_t, mask_t)

        # Convert to numpy
        img_arr = np.array(canvas_img, dtype=np.float32) / 255.0  # (H,W)
        mask_arr = np.array(canvas_mask, dtype=np.float32) / 255.0

        # Binarize mask: anything > 0 becomes 1.0
        mask_arr = (mask_arr > 0.5).astype(np.float32)

        # Add channel dimension → (1,H,W)
        img_arr = img_arr[np.newaxis, :, :]
        mask_arr = mask_arr[np.newaxis, :, :]

        x = torch.from_numpy(img_arr)             # float32, (1,H,W)
        y_mask = torch.from_numpy(mask_arr)       # float32, (1,H,W)
        y_cls = torch.tensor(label_idx, dtype=torch.long)

        return x, y_cls, y_mask


# ================================================================
# Model: shared encoder with class + mask heads
# ================================================================

class SmallConvNetMultiTask(nn.Module):
    """
    Tiny CNN that:
      - Produces class logits (for patch classification).
      - Produces a per-pixel mask (for segmentation).
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature extractor (same spirit as SmallConvNet)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 1/2

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 1/4
        )

        # Classification head: global avg pool -> FC
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),         # (N,16,1,1) -> (N,16)
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        # Segmentation head: upsample features back to input size
        # and use a small conv stack to produce 1-channel logits.
        self.seg_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        x: (N, C, H, W)
        Returns:
            cls_logits: (N, num_classes)
            seg_logits: (N, 1, H, W)
        """
        N, C, H, W = x.shape

        feats = self.features(x)              # (N,16,H/4,W/4)

        # ---- classification branch ----
        pooled = self.pool(feats)            # (N,16,1,1)
        cls_logits = self.classifier(pooled) # (N,num_classes)

        # ---- segmentation branch ----
        up = F.interpolate(
            feats,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )                                     # (N,16,H,W)
        seg_logits = self.seg_head(up)        # (N,1,H,W)

        return cls_logits, seg_logits


# ================================================================
# Training loop
# ================================================================

def runner_torch_dirty_train():
    print("=== TORCH MULTI-SCALE SEG TRAIN START ===")

    # 1) Build dataset
    dataset = ScorchMultiScaleSegDataset(
        annotations_subdir="training",
        images_subdir="training",
        fixed_width=FIXED_WIDTH,
        fixed_height=FIXED_HEIGHT,
        grayscale=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty — check your training annotations/images.")

    # Inspect one sample to get in_channels / shapes
    sample_x, sample_y_cls, sample_y_mask = dataset[0]
    in_channels = sample_x.shape[0]
    num_classes = len(CLASS_NAMES)

    print(f"[Train] in_channels={in_channels}, num_classes={num_classes}")
    print(f"[Train] class_names={CLASS_NAMES}")
    print(f"[Train] sample x shape={tuple(sample_x.shape)}, mask shape={tuple(sample_y_mask.shape)}")

    # 2) Model, loss, optimizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = SmallConvNetMultiTask(
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_mask = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) Simple "batch_size = 1" training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_mask_loss = 0.0
        correct = 0
        total = 0

        for i in range(len(dataset)):
            x, y_cls, y_mask = dataset[i]   # x: (1,H,W), y_mask: (1,H,W)

            x = x.unsqueeze(0).to(device)        # (1,1,H,W)
            y_cls = y_cls.unsqueeze(0).to(device)   # (1,)
            y_mask = y_mask.unsqueeze(0).to(device) # (1,1,H,W)

            optimizer.zero_grad()

            logits, seg_logits = model(x)

            loss_cls = criterion_cls(logits, y_cls)
            loss_mask = criterion_mask(seg_logits, y_mask)
            loss = loss_cls + LAMBDA_MASK * loss_mask

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cls_loss += loss_cls.item()
            running_mask_loss += loss_mask.item()

            # classification accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

        avg_loss = running_loss / total
        avg_cls_loss = running_cls_loss / total
        avg_mask_loss = running_mask_loss / total
        acc = correct / total if total > 0 else 0.0

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"loss={avg_loss:.4f} "
            f"(cls={avg_cls_loss:.4f}, mask={avg_mask_loss:.4f}), "
            f"acc={acc:.3f}"
        )

    # 4) Save model
    save_path = Path("hello_torch_dirty_multiscale_seg.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Train] Model saved to {save_path}")
    print("=== TORCH MULTI-SCALE SEG TRAIN DONE ===")


if __name__ == "__main__":
    runner_torch_dirty_train()
