# convolve_kernel_offset_test_b.py

from __future__ import annotations

from typing import List, Tuple

from image_tools.mask_loader import load_mask_white_xy_weights
from filesystem.file_utils import FileUtils

from image.bitmap import Bitmap

from image.convolve_padding_mode import (
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
    ConvolvePaddingMode,
)

# ------------------------------------------------------------
# Global test parameters (single configuration per run)
# ------------------------------------------------------------

STRIDE_H   = 1
STRIDE_V   = 1

DILATION_H = 1
DILATION_V = 1

MAX_OFFSET_X = 4
MAX_OFFSET_Y = 4

# Offsets to test
OFFSETS: List[Tuple[int, int]] = [
    (0, 0),
    (-4, -4),
    (4, -4),
    (-4, 4),
    (4, 4),
]

OUTPUT_SUBDIR = "kernel_offset_test_output_ii"

IMAGES_SUBDIR  = "training_images"
SOURCE_IMAGE   = "proto_cells_train_118.png"

KERNEL_SUBDIR  = "images/kernels"
KERNEL_NAME    = "soft_blur_kernel_3_3"


def _offset_tag(ox: int, oy: int) -> str:
    """
    Deterministic file-friendly tag: ox_p4 / ox_m4, oy_p0, etc.
    """
    def t(v: int) -> str:
        v = int(v)
        return f"p{v}" if v >= 0 else f"m{abs(v)}"
    return f"ox_{t(ox)}_oy_{t(oy)}"


def main() -> None:
    source_image = FileUtils.load_local_bitmap(IMAGES_SUBDIR, SOURCE_IMAGE)
    print(f"image => {source_image.width}x{source_image.height}")

    mask = load_mask_white_xy_weights(KERNEL_SUBDIR, KERNEL_NAME)
    kw = len(mask)
    kh = len(mask[0]) if kw > 0 else 0
    print(f"mask => {kw}x{kh}  ({KERNEL_NAME})")

    modes: List[Tuple[str, ConvolvePaddingMode]] = [
        ("s", ConvolvePaddingOffsetSame(max_offset_x=MAX_OFFSET_X, max_offset_y=MAX_OFFSET_Y)),
        ("v", ConvolvePaddingOffsetValid(max_offset_x=MAX_OFFSET_X, max_offset_y=MAX_OFFSET_Y)),
    ]

    for (mode_tag, padding_mode) in modes:
        print("\n============================================================")
        print("mode:", mode_tag, "padding:", Bitmap.padding_mode_string(padding_mode))
        print("============================================================")

        for (ox, oy) in OFFSETS:
            tag = _offset_tag(ox, oy)
            out_name = f"out_{KERNEL_NAME}_{mode_tag}_{tag}_convoluted"

            bmp_conv = source_image.convolve(
                mask=mask,
                offset_x=int(ox),
                offset_y=int(oy),
                stride_h=STRIDE_H,
                stride_v=STRIDE_V,
                dilation_h=DILATION_H,
                dilation_v=DILATION_V,
                padding_mode=padding_mode,
            )

            print(
                f"{mode_tag} {tag} => "
                f"bmp=({bmp_conv.width},{bmp_conv.height}) save={out_name}.png"
            )

            FileUtils.save_local_bitmap(bmp_conv, OUTPUT_SUBDIR, out_name, "png")

    print("\nDone.")


if __name__ == "__main__":
    main()
