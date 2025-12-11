# runner_scorch_deep_eval.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from filesystem.file_utils import FileUtils
from image.bitmap import Bitmap
from image.rgba import RGBA

from scorch_ext.data_loader import DataLoader
from scorch.scorch_tensor import ScorchTensor


# --- must match the training model architecture exactly ---
class SmallConvNet(nn.Module):
    """
    Same as in runner_scorch_train.py / runner_scorch_eval.py.
    Uses global avg pool so it works with variable-size patches.
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

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),   # (N,16,1,1) -> (N,16)
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# --- helper: draw a rectangle from (x,y,w,h) on a Bitmap ---
def draw_rect_on_bitmap(
    bmp: Bitmap,
    x0: int,
    y0: int,
    w: int,
    h: int,
    color: RGBA,
    thickness: int = 1,
) -> None:
    """
    Draw a rectangular frame on the bitmap in-place.

    (x0, y0) = top-left
    w, h     = width, height
    """
    if w <= 0 or h <= 0:
        return

    x1 = x0 + w - 1
    y1 = y0 + h - 1

    # Clamp to bitmap bounds
    x0 = max(0, min(x0, bmp.width - 1))
    x1 = max(0, min(x1, bmp.width - 1))
    y0 = max(0, min(y0, bmp.height - 1))
    y1 = max(0, min(y1, bmp.height - 1))

    if x0 >= x1 or y0 >= y1:
        return

    for t in range(thickness):
        # Top + bottom edges
        yy_top = max(y0 + t, 0)
        yy_bot = min(y1 - t, bmp.height - 1)
        for x in range(x0, x1 + 1):
            px_top = bmp.rgba[x][yy_top]
            px_bot = bmp.rgba[x][yy_bot]
            px_top.ri, px_top.gi, px_top.bi, px_top.ai = color.ri, color.gi, color.bi, 255
            px_bot.ri, px_bot.gi, px_bot.bi, px_bot.ai = color.ri, color.gi, color.bi, 255

        # Left + right edges
        xx_left = max(x0 + t, 0)
        xx_right = min(x1 - t, bmp.width - 1)
        for y in range(y0, y1 + 1):
            px_left  = bmp.rgba[xx_left][y]
            px_right = bmp.rgba[xx_right][y]
            px_left.ri, px_left.gi, px_left.bi, px_left.ai   = color.ri, color.gi, color.bi, 255
            px_right.ri, px_right.gi, px_right.bi, px_right.ai = color.ri, color.gi, color.bi, 255


def runner_scorch_deep_eval():
    print("=== SCORCH DEEP EVAL (SLIDING WINDOW) START ===")

    # --------------------------------------------------
    # 1) Dataset / file discovery (we mostly need images)
    # --------------------------------------------------
    loader = DataLoader(
        annotations_subdir="testing",
        images_subdir="testing",
    )

    # Same class order as training
    class_names = ["Red", "Blue", "Green"]
    num_classes = len(class_names)

    # Colors per class (for boxes)
    class_colors = {
        0: RGBA(255,   0,   0, 255),   # Red
        1: RGBA(  0, 255,   0, 255),   # Green
        2: RGBA(  0,   0, 255, 255),   # Blue
    }

    # --------------------------------------------------
    # 2) Model + weights
    # --------------------------------------------------
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    in_channels = 1  # trained in grayscale
    model = SmallConvNet(in_channels, num_classes).to(device)
    state_dict = torch.load("hello_torch_2.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --------------------------------------------------
    # 3) Sliding-window config
    # --------------------------------------------------
    # These are easy to tweak:
    PATCH_SIZE = 64      # window size (pixels)
    STRIDE     = 16      # step size (pixels)
    TOP_K_PER_CLASS = 5  # "N number of class matches" per class
    MIN_PROB   = 0.6     # ignore very low-confidence hits

    print(f"[DEEP EVAL] PATCH_SIZE={PATCH_SIZE}, STRIDE={STRIDE}, "
          f"TOP_K_PER_CLASS={TOP_K_PER_CLASS}, MIN_PROB={MIN_PROB}")

    for pair_index, pair in enumerate(loader):
        bmp = Bitmap.with_image(pair.image_path)
        H, W = bmp.height, bmp.width

        print(f"[DEEP EVAL] Pair {pair_index}: {pair.image_path} "
              f"(size={W}x{H})")

        candidates = []  # (prob, class_idx, x0, y0, w, h)

        # --------------------------------------------------
        # 4) Slide window across the whole image
        # --------------------------------------------------
        # We let Bitmap.crop handle off-edge clamping, so
        # we can run x=0..W, y=0..H and still get PATCH_SIZE x PATCH_SIZE.
        # To avoid too many windows, we stop at W,H plus one step.
        # --------------------------------------------------

        for y0 in range(0, H, STRIDE):
            for x0 in range(0, W, STRIDE):

                # Crop patch -> ScorchTensor -> Torch
                patch_tensor = ScorchTensor.from_bitmap_crop(
                    bmp=bmp,
                    x=x0,
                    y=y0,
                    width=PATCH_SIZE,
                    height=PATCH_SIZE,
                    name=None,
                    role="image_patch",
                    grayscale=True,
                )

                x = patch_tensor.to_torch().unsqueeze(0).to(device)  # (1,1,Hp,Wp)

                with torch.no_grad():
                    logits = model(x)                 # (1,num_classes)
                    probs = F.softmax(logits, dim=1)  # (1,num_classes)
                    prob_vals, pred_idxs = probs.max(dim=1)

                prob = float(prob_vals.item())
                c_idx = int(pred_idxs.item())

                if prob < MIN_PROB:
                    continue

                candidates.append((prob, c_idx, x0, y0, PATCH_SIZE, PATCH_SIZE))

        print(f"[DEEP EVAL]   Raw candidates above MIN_PROB: {len(candidates)}")

        # --------------------------------------------------
        # 5) Keep top-K per class (very crude detector)
        # --------------------------------------------------
        per_class = {i: [] for i in range(num_classes)}
        for prob, c_idx, x0, y0, w, h in candidates:
            per_class[c_idx].append((prob, c_idx, x0, y0, w, h))

        final_boxes = []
        for c_idx, cand_list in per_class.items():
            cand_list.sort(key=lambda t: t[0], reverse=True)
            top = cand_list[:TOP_K_PER_CLASS]
            final_boxes.extend(top)

            if top:
                print(f"    class={class_names[c_idx]}  kept={len(top)}  "
                      f"best_prob={top[0][0]:.3f}")

        # --------------------------------------------------
        # 6) Draw boxes on the original bitmap
        # --------------------------------------------------
        for prob, c_idx, x0, y0, w, h in final_boxes:
            color = class_colors.get(c_idx, RGBA(255, 255, 0, 255))
            draw_rect_on_bitmap(
                bmp=bmp,
                x0=x0,
                y0=y0,
                w=w,
                h=h,
                color=color,
                thickness=2,
            )

        # --------------------------------------------------
        # 7) Save the overlay
        # --------------------------------------------------
        out_name = f"deep_overlay_pair_{pair_index}.png"
        FileUtils.save_local_bitmap(
            bmp,
            "output_deep_eval",
            out_name,
        )
        print(f"[DEEP EVAL]   Saved overlay -> output_deep_eval/{out_name}")

    print("=== SCORCH DEEP EVAL DONE ===")


if __name__ == "__main__":
    runner_scorch_deep_eval()
