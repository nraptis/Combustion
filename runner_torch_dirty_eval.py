from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F

from scorch_ext.data_loader import DataLoader as ScorchLoader
from scorch_ext.data_loader import AnnotationImagePair
from scorch.scorch_tensor import ScorchTensor

# Reuse config + model from training
from runner_torch_dirty_train import (
    SmallConvNetMultiTask,
    CLASS_NAMES,
    FIXED_WIDTH,
    FIXED_HEIGHT,
)

# ================================================================
# Eval config
# ================================================================

MODEL_PATH = Path("hello_torch_dirty_multiscale_seg.pth")

# Sizes of square windows to try on the atlas (in original atlas pixels)
# (Adjust to taste; keep them reasonable relative to your circles)
SLIDING_WINDOW_SIZES = list(range(24, 104, 4))  # 24, 28, ..., 100

# Stride (in pixels) for the sliding window (on the atlas)
SLIDING_STRIDE = 4  # 2 or 4 usually fine

# Probability thresholds & IoU-ish scoring
MIN_CLASS_PROB = 0.6     # minimum softmax prob for best class
MIN_MASK_FRACTION = 0.2  # how "filled" a patch should be
MIN_SCORE = 0.5          # combined score threshold (prob * mask_fraction)

# NMS-ish overlap control (circle-based like your sketch)
# If distance^2 <= (r1 + r2)^2 * NMS_FACTOR we consider it overlapping and drop the weaker.
NMS_FACTOR = 0.8

# Testing split (can temporarily point this at "training" for sanity checks)
EVAL_ANNOTATIONS_SUBDIR = "testing"
EVAL_IMAGES_SUBDIR = "testing"

OUTPUT_DIR = Path("testing_detections")


# ================================================================
# IOUKey / detection representation
# ================================================================

@dataclass(frozen=True)
class IOUKey:
    """
    Represents one detection candidate on the atlas.

    (x, y) is the *top-left* corner of a square patch of size `size`
    in the atlas coordinate system. class_idx is an index into CLASS_NAMES.
    """
    size: int
    class_idx: int
    x: int
    y: int

    @property
    def center(self) -> Tuple[float, float]:
        cx = self.x + self.size / 2.0
        cy = self.y + self.size / 2.0
        return cx, cy

    @property
    def radius(self) -> float:
        # interpret detection as a circle inscribed in the square
        return self.size / 2.0


@dataclass
class Detection:
    key: IOUKey
    score: float  # our "IoU-ish" score: prob * mask_fraction


# ================================================================
# Helpers
# ================================================================

def atlas_from_pair(pair: AnnotationImagePair) -> Image.Image:
    """
    Get a PIL image for the atlas from an AnnotationImagePair.

    Adjust this depending on how your AnnotationImagePair exposes the
    underlying Bitmap / PIL / path.

    This version assumes:
        - pair.image_bitmap is your Bitmap wrapper.
        - Bitmap has a to_image() -> PIL.Image.Image method.

    If instead you have pair.image_path or pair.image (PIL), tweak this.
    """
    # ---- OPTION A: Bitmap wrapper ----
    # bmp: Bitmap = pair.image_bitmap
    # return bmp.to_image()

    # ---- OPTION B: direct PIL stored on the pair ----
    # return pair.image

    # ---- OPTION C: path stored on pair ----
    # return Image.open(pair.image_path).convert("L")

    # For now, we try a sensible default and let you tweak:
    if hasattr(pair, "image_bitmap"):
        return pair.image_bitmap.to_image()
    elif hasattr(pair, "image"):
        img = pair.image
        return img.convert("L") if img.mode != "L" else img
    elif hasattr(pair, "image_path"):
        img = Image.open(pair.image_path)
        return img.convert("L") if img.mode != "L" else img
    else:
        raise AttributeError(
            "atlas_from_pair: cannot find image on AnnotationImagePair "
            "(expected image_bitmap / image / image_path)."
        )


def sliding_window_patches(
    atlas: Image.Image,
    size: int,
    stride: int,
) -> List[Tuple[int, int, Image.Image]]:
    """
    Iterate over square patches of `size x size` in `atlas` with given stride.

    Returns list of (x, y, patch_image) where (x, y) is top-left in atlas coords.
    """
    w, h = atlas.size
    patches: List[Tuple[int, int, Image.Image]] = []

    max_x = max(0, w - size)
    max_y = max(0, h - size)

    y = 0
    while y <= max_y:
        x = 0
        while x <= max_x:
            patch = atlas.crop((x, y, x + size, y + size))
            patches.append((x, y, patch))
            x += stride
        y += stride

    return patches


def prepare_model_input_from_patch(
    patch: Image.Image,
    fixed_width: int = FIXED_WIDTH,
    fixed_height: int = FIXED_HEIGHT,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Given a square atlas patch (PIL), resize to (fixed_width, fixed_height)
    and convert to the (1, 1, H, W) float32 tensor in [0,1].
    """
    if patch.mode != "L":
        patch = patch.convert("L")

    resized = patch.resize((fixed_width, fixed_height), resample=Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0  # (H,W)
    arr = arr[np.newaxis, :, :]  # (1,H,W)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,1,H,W)
    return x


MAX_WINDOW_SIZE = max(SLIDING_WINDOW_SIZES)

def compute_patch_score(
    logits: torch.Tensor,
    seg_logits: torch.Tensor,
    window_size: int,
) -> Tuple[int, float, float, float]:
    probs = F.softmax(logits, dim=1)
    best_prob, best_idx = probs.max(dim=1)

    best_class_idx = int(best_idx.item())
    best_class_prob = float(best_prob.item())

    seg_probs = torch.sigmoid(seg_logits)[0, 0, :, :]
    mask_bin = (seg_probs > 0.5).float()
    mask_fraction = float(mask_bin.mean().item())

    # size_bias in [0.5, 1.0] – bigger windows slightly preferred
    size_norm = window_size / float(MAX_WINDOW_SIZE)
    size_bias = 0.5 + 0.5 * size_norm

    score = best_class_prob * mask_fraction * size_bias

    return best_class_idx, best_class_prob, mask_fraction, score


def non_max_suppression_circle(
    detections: List[Detection],
    nms_factor: float = NMS_FACTOR,
) -> List[Detection]:
    """
    NMS-style pruning using your circle overlap idea:

        thresh = (r1 + r2)^2 * nms_factor
        if (dx^2 + dy^2) <= thresh  -> overlapping -> keep only higher score.

    Detections are assumed to be from the same atlas.

    Returns a pruned list of detections.
    """
    if not detections:
        return []

    # Sort by descending score
    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []

    for det in detections:
        cx1, cy1 = det.key.center
        r1 = det.key.radius

        discard = False
        for kept_det in kept:
            cx2, cy2 = kept_det.key.center
            r2 = kept_det.key.radius

            dx = cx1 - cx2
            dy = cy1 - cy2
            dist2 = dx * dx + dy * dy
            thresh = (r1 + r2) * (r1 + r2) * nms_factor

            if dist2 <= thresh:
                # Overlaps with a better detection
                discard = True
                break

        if not discard:
            kept.append(det)

    return kept


def draw_detections_on_atlas(
    atlas: Image.Image,
    detections: List[Detection],
    draw_labels: bool = True,
) -> Image.Image:
    """
    Draw square boxes for detections on a copy of the atlas.
    Optionally add class-name text at the top-left of each box.
    """
    out = atlas.convert("RGB")
    draw = ImageDraw.Draw(out)

    # Pick a font if available; fallback gracefully.
    font = None
    if draw_labels:
        try:
            # You can change this path to a specific TTF if you like.
            font = ImageFont.load_default()
        except Exception:
            font = None

    for det in detections:
        k = det.key
        x0, y0 = k.x, k.y
        x1, y1 = x0 + k.size, y0 + k.size

        # Box
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=2)

        if draw_labels:
            class_name = CLASS_NAMES[k.class_idx]
            label = f"{class_name}"
            if font is not None:
                draw.text((x0 + 2, y0 + 2), label, fill=(255, 255, 0), font=font)
            else:
                draw.text((x0 + 2, y0 + 2), label, fill=(255, 255, 0))

    return out


# ================================================================
# Main: per-pair detection + box drawing
# ================================================================

def run_detection_on_pair(
    pair: AnnotationImagePair,
    model: SmallConvNetMultiTask,
    device: str,
) -> Tuple[Image.Image, List[Detection]]:
    """
    Core detection routine for a single AnnotationImagePair.

    Steps:
      - Get full atlas (PIL) from the pair.
      - For each sliding window size:
        - Slide over atlas with SLIDING_STRIDE.
        - For each patch:
          - Resize to Fixed canvas, run model.
          - Compute score = best_prob * mask_fraction.
          - If above thresholds, store Detection in list.
      - Run circle-based NMS on the detections.
      - Draw resulting detections on atlas.

    Returns:
      (output_image_with_boxes, kept_detections)
    """
    atlas = atlas_from_pair(pair)
    w, h = atlas.size
    print(f"[Detect] Atlas size = {w}x{h}")

    detections: List[Detection] = []

    with torch.no_grad():
        for size in SLIDING_WINDOW_SIZES:
            if size > w or size > h:
                continue  # skip sizes larger than atlas

            patches = sliding_window_patches(
                atlas=atlas,
                size=size,
                stride=SLIDING_STRIDE,
            )

            print(f"[Detect] Size={size}, patches={len(patches)}")

            for (x, y, patch) in patches:
                x_input = prepare_model_input_from_patch(
                    patch=patch,
                    fixed_width=FIXED_WIDTH,
                    fixed_height=FIXED_HEIGHT,
                    device=device,
                )

                logits, seg_logits = model(x_input)

                class_idx, class_prob, mask_fraction, score = compute_patch_score(
                    logits=logits,
                    seg_logits=seg_logits,
                    window_size=size,
                )

                # crude "IoU-ish" score: how confident, and how foreground-y
                # score = class_prob * mask_fraction

                if class_prob < MIN_CLASS_PROB:
                    continue
                if mask_fraction < MIN_MASK_FRACTION:
                    continue
                if score < MIN_SCORE:
                    continue

                key = IOUKey(
                    size=size,
                    class_idx=class_idx,
                    x=x,
                    y=y,
                )
                detections.append(Detection(key=key, score=score))

    print(f"[Detect] Raw detections (before NMS): {len(detections)}")

    kept = non_max_suppression_circle(detections, nms_factor=NMS_FACTOR)
    print(f"[Detect] Kept detections (after NMS): {len(kept)}")

    out_img = draw_detections_on_atlas(atlas, kept, draw_labels=True)
    return out_img, kept


# ================================================================
# Runner
# ================================================================

def runner_torch_dirty_eval() -> None:
    print("=== TORCH MULTI-SCALE SLIDING EVAL START ===")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[Eval] Using device: {device}")

    # 1) Rebuild model + load weights
    in_channels = 1  # you trained grayscale
    num_classes = len(CLASS_NAMES)

    model = SmallConvNetMultiTask(
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[Eval] Loaded model from {MODEL_PATH}")
    print(f"[Eval] CLASS_NAMES = {CLASS_NAMES}")
    print(f"[Eval] SLIDING_WINDOW_SIZES = {SLIDING_WINDOW_SIZES}")

    # 2) Build loader over testing pairs
    loader = ScorchLoader(
        annotations_subdir=EVAL_ANNOTATIONS_SUBDIR,
        images_subdir=EVAL_IMAGES_SUBDIR,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, pair in enumerate(loader):
        print(f"\n[Eval] Processing pair {idx} ...")

        out_img, kept = run_detection_on_pair(pair, model, device=device)

        # Try to build a reasonable output filename from the pair
        base_name = f"pair_{idx:03d}"
        if hasattr(pair, "image_path"):
            base_name = Path(pair.image_path).stem
        elif hasattr(pair, "name"):
            base_name = str(pair.name)

        out_path = OUTPUT_DIR / f"{base_name}_det.png"
        out_img.save(out_path)
        print(f"[Eval] Saved detection visualization to: {out_path}")
        print(f"[Eval] Detections:")
        for d in kept:
            k = d.key
            print(
                f"    class={CLASS_NAMES[k.class_idx]:>5}, "
                f"score={d.score:.3f}, size={k.size}, x={k.x}, y={k.y}"
            )

    print("\n=== TORCH MULTI-SCALE SLIDING EVAL DONE ===")


if __name__ == "__main__":
    runner_torch_dirty_eval()
