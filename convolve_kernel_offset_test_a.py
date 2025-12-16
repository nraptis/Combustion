# kernel_offset_sweep_test.py

from __future__ import annotations

from typing import List, Tuple

from image_tools.mask_loader import load_mask_white_xy_weights
from filesystem.file_utils import FileUtils

from image.bitmap import Bitmap
from image.rgba import RGBA

from labels.image_annotation_document import ImageAnnotationDocument

from image.convolve_kernel_alignment import ConvolveKernelAlignment
from image.convolve_padding_mode import (
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
    ConvolvePaddingMode,
)

# ------------------------------------------------------------
# Global test parameters (single configuration per run)
# ------------------------------------------------------------

STRIDE_H   = 1
STRIDE_V   = 3

DILATION_H = 1
DILATION_V = 4

MAX_OFFSET_X = 2
MAX_OFFSET_Y = 40

# Offsets to test (you can add diagonals if you want)
OFFSETS: List[Tuple[int, int]] = [

    (2, -4)
    
    
]

"""
    ( 0,  0),
    ( 1,  0),
    (-1,  0),
    ( 0,  1),
    ( 0, -1),
    ( 4,  0),
    (-4,  0),
    ( 0,  4),
    ( 0, -4),
    """

OUTPUT_SUBDIR = "kernel_offset_test_output"

ANNOTATIONS_SUBDIR = "training_labels"
IMAGES_SUBDIR      = "training_images"

SOURCE_IMAGE       = "proto_cells_train_118.png"
SOURCE_ANNOTATIONS = "proto_cells_train_118_annotations.json"

KERNEL_SUBDIR      = "images/kernels"
KERNEL_NAME        = "kernel_9_9_c_c"

# Alignment is ONLY for pixel-bag/document mapping, not bitmap convolve.
# Since this kernel is center-labeled, keep it CENTER for this test.
DOC_ALIGNMENT = ConvolveKernelAlignment.CENTER


def _offset_tag(ox: int, oy: int) -> str:
    """
    Deterministic file-friendly tag: ox_p1 / ox_m4, oy_p0, etc.
    """
    def t(v: int) -> str:
        return f"p{v}" if v >= 0 else f"m{abs(v)}"
    return f"ox_{t(int(ox))}_oy_{t(int(oy))}"


def _save_triplet(
    *,
    base: str,
    source_image: Bitmap,
    convoluted: Bitmap,
    the_mask: Bitmap | None,
) -> None:
    """
    Saves in alphabetical order:
      original
      convoluted
      the_mask
    """
    FileUtils.save_local_bitmap(source_image, OUTPUT_SUBDIR, base + "_original", "png")
    FileUtils.save_local_bitmap(convoluted,   OUTPUT_SUBDIR, base + "_convoluted", "png")
    if the_mask is not None:
        FileUtils.save_local_bitmap(the_mask, OUTPUT_SUBDIR, base + "_the_mask", "png")


def main() -> None:
    # Load once
    source_image = FileUtils.load_local_bitmap(IMAGES_SUBDIR, SOURCE_IMAGE)
    print(f"image => {source_image.width}x{source_image.height}")

    document = ImageAnnotationDocument.from_local_file(
        subdirectory=ANNOTATIONS_SUBDIR,
        name=SOURCE_ANNOTATIONS,
        extension=None,
    )
    print(document)

    mask = load_mask_white_xy_weights(KERNEL_SUBDIR, KERNEL_NAME)
    print(f"mask => {len(mask)}x{len(mask[0]) if len(mask)>0 else 0}  ({KERNEL_NAME})")

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
            base = f"out_{KERNEL_NAME}_{mode_tag}_{tag}"

            # ------------------------------------------------------------
            # Bitmap convolve (alignment-agnostic)
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # Frame anchor print (THIS is what should shift with offset)
            # ------------------------------------------------------------
            ax, ay, out_w, out_h = Bitmap.convolve_frame_mask(
                image_width=source_image.width,
                image_height=source_image.height,
                mask=mask,
                offset_x=int(ox),
                offset_y=int(oy),
                stride_h=STRIDE_H,
                stride_v=STRIDE_V,
                dilation_h=DILATION_H,
                dilation_v=DILATION_V,
                kernel_alignment=DOC_ALIGNMENT,  # only affects anchor reporting
                padding_mode=padding_mode,
            )

            print(
                f"{mode_tag} {tag} => "
                f"anchor_src=({ax},{ay}) out=({out_w},{out_h}) "
                f"bmp=({bmp_conv.width},{bmp_conv.height})"
            )

            # ------------------------------------------------------------
            # Transform document into convolved output space (pixel-bag mapping)
            # ------------------------------------------------------------
            doc_conv = document.transformed_with_convolve(
                mask=mask,
                offset_x=int(ox),
                offset_y=int(oy),
                stride_h=STRIDE_H,
                stride_v=STRIDE_V,
                dilation_h=DILATION_H,
                dilation_v=DILATION_V,
                kernel_alignment=DOC_ALIGNMENT,
                padding_mode=padding_mode,
            )

            # Render all labels into a mask bitmap using the NEW output size
            the_mask = doc_conv.to_bitmap()  # your defaults should be fine

            # ------------------------------------------------------------
            # Save triplet (alphabetical)
            # ------------------------------------------------------------
            _save_triplet(
                base=base,
                source_image=source_image,
                convoluted=bmp_conv,
                the_mask=the_mask,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
