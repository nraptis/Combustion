from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# User knobs (as requested)
# -----------------------------
REFERENCE_IMAGE_PATH = "reference_cat.png"
SCALE_FACTOR = 4  # 32 -> 128
OUTPUT_DIRECTORY = "training_video_frames"
OUTPUT_STEM = "video_frame_"  # saves as video_frame_00000.png


# -----------------------------
# Fixed layout from your screenshot
# -----------------------------
CANVAS_W = 720
CANVAS_H = 780   # was 504

MARGIN = 20
TOP_BAND_H = 80

TILE = 128
GAP = 10

GRID_COLS = 5
GRID_ROWS = 5     # was 3

# Colors (match vibe; tweak freely)
BG_COLOR = (0, 0, 0)
TILE_BG = (153, 0, 0)        # dark red
TILE_TEXT = (255, 255, 255)
HEADER_TEXT = (255, 255, 255)


# -----------------------------
# Helpers: image conversions
# -----------------------------
def _pil_to_tensor_chw_01(pil: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor (3,H,W) in [0,1]."""
    pil = pil.convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1)        # (3,H,W)
    return t


def _tensor_chw_01_to_pil(t: torch.Tensor) -> Image.Image:
    """torch float tensor (3,H,W) in [0,1] -> PIL RGB."""
    t = t.detach().clamp(0, 1).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # (H,W,3)
    return Image.fromarray(arr, mode="RGB")


def dumb_upscale_tensor_nearest(t_chw: torch.Tensor, scale: int) -> torch.Tensor:
    """Nearest-neighbor 'blocky' upscale for CHW tensor in [0,1]."""
    if t_chw.ndim != 3:
        raise ValueError("Expected CHW tensor")
    c, h, w = t_chw.shape
    t_bchw = t_chw.unsqueeze(0)  # (1,C,H,W)
    up = F.interpolate(t_bchw, size=(h * scale, w * scale), mode="nearest")
    return up.squeeze(0)


def load_reference_tile_128(path: str, scale: int = SCALE_FACTOR) -> Image.Image:
    """Loads reference 32x32 PNG and returns a 128x128 PIL image (blocky)."""
    pil = Image.open(path).convert("RGB")
    if pil.size != (32, 32):
        raise ValueError(f"Reference must be 32x32, got {pil.size}")
    t = _pil_to_tensor_chw_01(pil)
    up = dumb_upscale_tensor_nearest(t, scale=scale)  # (3,128,128)
    return _tensor_chw_01_to_pil(up)

def build_index_to_pos(grid_rows: int, grid_cols: int, ref_row: int, ref_col: int):
    """
    Returns mapping for 24 kernel tiles in row-major order, skipping the reference cell.
    Keys: 0..23
    Values: (row, col)
    """
    out = {}
    k = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if r == ref_row and c == ref_col:
                continue
            out[k] = (r, c)
            k += 1
            if k == 24:
                return out
    raise RuntimeError("Grid too small to fit 24 tiles + reference")

# -----------------------------
# Layout mapping
# -----------------------------
def tile_xy(row: int, col: int) -> Tuple[int, int]:
    """Top-left pixel of a tile in the 5x3 grid."""
    x = MARGIN + col * (TILE + GAP)
    y = TOP_BAND_H + row * (TILE + GAP)
    return x, y


# Center reference (G) tile position: row 1, col 2 (0-indexed)
REF_ROW = 2       # middle row (0..4)
REF_COL = 2       # middle col (0..4)

# The 14 dynamic tiles (0..13) skip the center tile.
# This matches your screenshot numbering layout.
INDEX_TO_POS = build_index_to_pos(GRID_ROWS, GRID_COLS, REF_ROW, REF_COL)

def _get_font(size: int, bold: bool = False):
    from PIL import ImageFont
    import os
    import PIL

    candidates = []

    # Pillow-bundled fonts (best if present)
    pil_dir = os.path.dirname(PIL.__file__)
    candidates.append(os.path.join(pil_dir, "fonts", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))

    # Common macOS font locations
    candidates += [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Verdana.ttf",
    ]

    for path in candidates:
        if path and os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=size)
                return font
            except Exception as e:
                print(f"Font load failed: {path} -> {e}")

    print("WARNING: No TTF font found, using default bitmap font (size will NOT scale).")
    return ImageFont.load_default()


def render_training_frame(
    frame_index: int,
    reference_tile_128: Image.Image,
    tiles: Optional[Dict[int, Image.Image]] = None,
    accuracy: Optional[float] = None,         # 0..1
    epoch: Optional[int] = None,
    epochs: Optional[int] = None,
    progress: Optional[float] = None,         # 0..1 (e.g. percent through epoch)
) -> Image.Image:
    """
    tiles: dict mapping tile_index -> PIL image (128x128) OR None to use red placeholders.
          tile_index range is 0..13 (14 tiles). Reference tile is separate.

    Returns a PIL RGB frame sized 720x504.
    """
    frame = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(frame)

    # Header text
    header_font = _get_font(34)

    left_txt = ""
    mid_txt = ""
    right_txt = ""

    if accuracy is not None:
        left_txt = f"{int(round(accuracy * 100)):03d}% Accuracy"
    if epoch is not None and epochs is not None:
        epoch_w = len(str(epochs))      # number of digits needed
        mid_txt = f"Epoch {epoch:0{epoch_w}d}/{epochs}"
    if progress is not None:
        pct = int(round(progress * 100))
        right_txt = f"{pct:03d}% (Epoch)"

        

    # Place header (fonts can change; positions are stable)
    if left_txt:
        draw.text((MARGIN, 24), left_txt, fill=HEADER_TEXT, font=header_font)

    if mid_txt:
        draw.text((282, 24), mid_txt, fill=HEADER_TEXT, font=header_font)

    if right_txt:
        draw.text((500, 24), right_txt, fill=HEADER_TEXT, font=header_font)

    # Draw grid placeholders
    tile_font = _get_font(56)

    # Draw all 15 slots as red tiles first
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x, y = tile_xy(r, c)
            draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], fill=TILE_BG)

    # Paste reference tile (G slot)
    rx, ry = tile_xy(REF_ROW, REF_COL)
    frame.paste(reference_tile_128, (rx, ry))

    # Paste dynamic tiles or draw numbers
    tiles = tiles or {}
    for idx, (r, c) in INDEX_TO_POS.items():
        x, y = tile_xy(r, c)
        if idx in tiles and tiles[idx] is not None:
            img = tiles[idx].convert("RGB")
            if img.size != (TILE, TILE):
                img = img.resize((TILE, TILE), resample=Image.NEAREST)
            frame.paste(img, (x, y))
        else:
            # Draw index label as placeholder (like screenshot)
            label = str(idx)
            tw = draw.textlength(label, font=tile_font)
            draw.text((x + (TILE - tw) / 2, y + 32), label, fill=TILE_TEXT, font=tile_font)

    # Optional: draw a subtle "G" label on top of the reference tile (if you want)
    # draw.text((rx + 10, ry + 10), "G", fill=TILE_TEXT, font=_get_font(36))

    return frame


def save_training_frame(img: Image.Image, frame_index: int, output_dir: str = OUTPUT_DIRECTORY) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{OUTPUT_STEM}{frame_index:05d}.png")
    img.save(path)
    return path


def demo_generate_one_frame():
    ref = load_reference_tile_128(REFERENCE_IMAGE_PATH, scale=SCALE_FACTOR)
    frame = render_training_frame(
        frame_index=0,
        reference_tile_128=ref,
        tiles=None,
        accuracy=0.55,
        epoch=5,
        epochs=10,
        progress=0.44,
    )
    out = save_training_frame(frame, 0)
    print("Wrote:", out)


if __name__ == "__main__":
    demo_generate_one_frame()
