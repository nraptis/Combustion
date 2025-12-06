# runner_scorch_eval.py

from __future__ import annotations

import torch
import torch.nn as nn

from filesystem.file_utils import FileUtils
from image.bitmap import Bitmap
from image.rgba import RGBA

from scorch.data_loader import DataLoader
from scorch.tensor_load_helpers import image_and_mask_from_pixel_bag
from scorch.scorch_tensor import ScorchTensor


# --- must match the training model architecture exactly ---
class SmallConvNet(nn.Module):
    """
    Same as in runner_scorch_train.py, but redefined here for loading.
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


# --- helper: draw a rectangle on a Bitmap using a PixelBag frame ---
def draw_frame_on_bitmap(bmp: Bitmap, bag, color: RGBA, thickness: int = 1) -> None:
    """
    Draw a rectangular frame around bag.frame on the bitmap in-place.
    bag.frame = (x0, y0, w, h)
    """
    x0, y0, w, h = bag.frame
    if w == 0 or h == 0:
        return

    x1 = x0 + w - 1
    y1 = y0 + h - 1

    # Clamp to bitmap bounds
    x0 = max(0, min(x0, bmp.width - 1))
    x1 = max(0, min(x1, bmp.width - 1))
    y0 = max(0, min(y0, bmp.height - 1))
    y1 = max(0, min(y1, bmp.height - 1))

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


def runner_scorch_eval():
    print("=== SCORCH EVAL START ===")

    # 1) Rebuild dataset-ish view using DataLoader
    loader = DataLoader(
        annotations_subdir="testing",
        images_subdir="testing",
    )

    # We’ll reconstruct the same class mapping that training used.
    # Since you're using the same data, the order should be identical.
    class_names = ["Red", "Blue", "Green"]   # quick & dirty; or infer dynamically
    num_classes = len(class_names)
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    # 2) Build model and load weights
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Need in_channels = 1 (we trained grayscale), so hard-code for now.
    in_channels = 1
    model = SmallConvNet(in_channels, num_classes).to(device)
    state_dict = torch.load("hello_torch.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Colors: correct = green frame, wrong = red frame
    color_correct = RGBA(0, 255, 0, 255)
    color_wrong   = RGBA(255, 0, 0, 255)

    total = 0
    correct = 0

    for pair_index, pair in enumerate(loader):
        bmp = Bitmap.with_image(pair.image_path)
        doc = pair.document

        print(f"[EVAL] Pair {pair_index}: {pair.image_path}")

        for label in doc.data.labels:
            bag = label.pixel_bag
            true_name = label.name

            # 3) crop patch using the same helper as training
            img_patch, _mask_patch = image_and_mask_from_pixel_bag(
                bmp=bmp,
                bag=bag,
                name=label.name,
                grayscale=True,
            )

            # Convert to torch
            x = img_patch.to_torch()          # (C,H,W)
            x = x.unsqueeze(0).to(device)     # (1,C,H,W)

            with torch.no_grad():
                logits = model(x)
                pred_idx = int(logits.argmax(dim=1).item())

            pred_name = class_names[pred_idx]

            is_correct = (pred_name == true_name)
            total += 1
            correct += int(is_correct)

            # 4) draw a box on the original bitmap
            draw_frame_on_bitmap(
                bmp=bmp,
                bag=bag,
                color=color_correct if is_correct else color_wrong,
                thickness=2,
            )

            print(
                f"  label={true_name:5s}  pred={pred_name:5s}  "
                f"{'OK' if is_correct else 'MISS'}"
            )

        # 5) save overlay for this whole image
        FileUtils.save_local_bitmap(
            bmp,
            "output_eval",
            f"overlay_pair_{pair_index}.png",
        )

    acc = correct / total if total > 0 else 0.0
    print(f"[EVAL] total={total}, correct={correct}, acc={acc:.3f}")
    print("=== SCORCH EVAL DONE ===")


if __name__ == "__main__":
    runner_scorch_eval()
