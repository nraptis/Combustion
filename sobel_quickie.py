# sobel_quickie.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from filesystem.file_io import FileIO
from filesystem.file_utils import FileUtils


# ------------------------------------------------------------
# Core Sobel on a Pillow image
# ------------------------------------------------------------
def _compute_sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """
    gray: float32 array in [0, 1], shape (H, W)
    returns: float32 array in [0, 1], shape (H, W)
    """
    # Classic 3x3 Sobel kernels
    kx = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ky = np.array(
        [
            [-1.0, -2.0, -1.0],
            [0.0,  0.0,  0.0],
            [1.0,  2.0,  1.0],
        ],
        dtype=np.float32,
    )

    h, w = gray.shape
    # Pad with edge values so borders behave nicely
    padded = np.pad(gray, pad_width=1, mode="edge")

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)

    # Naive convolution; fine for a handful of images
    for y in range(h):
        ys = y
        for x in range(w):
            xs = x
            patch = padded[ys:ys + 3, xs:xs + 3]
            gx[y, x] = np.sum(patch * kx)
            gy[y, x] = np.sum(patch * ky)

    mag = np.hypot(gx, gy)  # sqrt(gx^2 + gy^2)

    # Normalize to [0, 1] for saving
    mag_min = float(mag.min())
    mag_max = float(mag.max())
    if mag_max > mag_min:
        mag = (mag - mag_min) / (mag_max - mag_min)
    else:
        mag[:] = 0.0

    return mag


def sobel_image_pillow(img: Image.Image) -> Image.Image:
    """
    Take a Pillow image, convert to grayscale, apply Sobel,
    and return a new Pillow image (mode "L") with edge magnitude.
    """
    gray_img = img.convert("L")
    gray = np.asarray(gray_img, dtype=np.float32) / 255.0
    mag = _compute_sobel_magnitude(gray)
    mag_uint8 = (np.clip(mag, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(mag_uint8, mode="L")


# ------------------------------------------------------------
# File / folder plumbing using FileIO + FileUtils
# ------------------------------------------------------------
def _iter_input_files() -> Iterable[Path]:
    """
    Yield all files in the local 'input' directory (non-recursive).
    Uses FileIO.get_all_files_local("input").
    """
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for p in FileIO.get_all_files_local("input"):
        if p.suffix.lower() in exts:
            yield p


def _output_path_for(input_path: Path) -> Path:
    """
    Given ROOT/input/asdf.png -> ROOT/output/asdf_sobel.png
    """
    out_dir = FileIO.local_directory("output")
    stem = input_path.stem
    out_name = f"{stem}_sobel.png"
    return (out_dir / out_name).resolve()


def process_all() -> None:
    """
    Walk 'input/' folder, run Sobel on each image,
    save to 'output/' with '_sobel' suffix.
    """
    out_dir = FileIO.local_directory("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(_iter_input_files())
    if not files:
        print("[sobel_quickie] No input images found in 'input/'")
        return

    print(f"[sobel_quickie] Found {len(files)} input image(s).")

    for idx, in_path in enumerate(files, start=1):
        print(f"[sobel_quickie] [{idx}/{len(files)}] {in_path.name} ...", end="", flush=True)
        # Load via FileUtils so we stay within your filesystem stack
        img = FileUtils.load_image(in_path)
        sobel_img = sobel_image_pillow(img)

        out_path = _output_path_for(in_path)
        FileUtils.save_image(sobel_img, out_path)
        print(f" -> {out_path.name}")

    print("[sobel_quickie] Done.")


if __name__ == "__main__":
    process_all()
