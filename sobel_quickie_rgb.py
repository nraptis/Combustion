# sobel_quickie_rgb.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from filesystem.file_io import FileIO
from filesystem.file_utils import FileUtils


# ------------------------------------------------------------
# Core Sobel on a single 2D channel
# ------------------------------------------------------------
def _compute_sobel_channel(gray: np.ndarray) -> np.ndarray:
    """
    gray: float32 array in [0, 1], shape (H, W)
    returns: float32 array in [0, 1], shape (H, W)
    """
    # 3x3 Sobel kernels
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
    padded = np.pad(gray, pad_width=1, mode="edge")

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)

    # Naive convolution (fine for small-ish images / batch)
    for y in range(h):
        ys = y
        for x in range(w):
            xs = x
            patch = padded[ys:ys + 3, xs:xs + 3]
            gx[y, x] = np.sum(patch * kx)
            gy[y, x] = np.sum(patch * ky)

    mag = np.hypot(gx, gy)

    # Normalize to [0,1]
    mag_min = float(mag.min())
    mag_max = float(mag.max())
    if mag_max > mag_min:
        mag = (mag - mag_min) / (mag_max - mag_min)
    else:
        mag[:] = 0.0

    return mag


def sobel_image_rgb_pillow(img: Image.Image) -> Image.Image:
    """
    Take a Pillow image, convert to RGB, apply Sobel per channel,
    recombine into an RGB edge image, and return a Pillow Image.
    """
    rgb_img = img.convert("RGB")
    arr = np.asarray(rgb_img, dtype=np.float32) / 255.0  # (H, W, 3)
    h, w, c = arr.shape
    assert c == 3

    out = np.zeros_like(arr, dtype=np.float32)

    for ch in range(3):
        channel = arr[..., ch]
        sobel_ch = _compute_sobel_channel(channel)
        out[..., ch] = sobel_ch

    out_uint8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(out_uint8, mode="RGB")


# ------------------------------------------------------------
# File / folder plumbing using FileIO + FileUtils
# ------------------------------------------------------------
def _iter_input_files() -> Iterable[Path]:
    """
    Yield all image files in the local 'input' directory (non-recursive).
    """
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for p in FileIO.get_all_files_local("input"):
        if p.suffix.lower() in exts:
            yield p


def _output_path_for(input_path: Path) -> Path:
    """
    Given ROOT/input/blah.png -> ROOT/output/rgb/blah_sobel_rgb.png
    """
    out_dir = FileIO.local_directory("output/rgb")
    out_name = f"{input_path.stem}_sobel_rgb.png"
    return (out_dir / out_name).resolve()


def process_all_rgb() -> None:
    """
    Walk 'input/' folder, run RGB Sobel on each image,
    save to 'output/rgb/' with '_sobel_rgb' suffix.
    """
    out_dir = FileIO.local_directory("output/rgb")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(_iter_input_files())
    if not files:
        print("[sobel_quickie_rgb] No input images found in 'input/'")
        return

    print(f"[sobel_quickie_rgb] Found {len(files)} input image(s).")

    for idx, in_path in enumerate(files, start=1):
        print(f"[sobel_quickie_rgb] [{idx}/{len(files)}] {in_path.name} ...", end="", flush=True)

        img = FileUtils.load_image(in_path)
        sobel_rgb_img = sobel_image_rgb_pillow(img)

        out_path = _output_path_for(in_path)
        FileUtils.save_image(sobel_rgb_img, out_path)
        print(f" -> {out_path.relative_to(FileIO.local_directory())}")

    print("[sobel_quickie_rgb] Done.")


if __name__ == "__main__":
    process_all_rgb()
