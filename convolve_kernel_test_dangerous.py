from __future__ import annotations

from typing import List

from image_tools.mask_loader import load_mask_white_xy_weights
from filesystem.file_utils import FileUtils

from labels.image_annotation_document import ImageAnnotationDocument

from image.convolve_kernel_alignment import ConvolveKernelAlignment
from image.convolve_padding_mode import (
    ConvolvePaddingSame,
    ConvolvePaddingValid,
    ConvolvePaddingMode,
)

# ------------------------------------------------------------
# Global test parameters (single configuration per run)
# ------------------------------------------------------------

STRIDE_H   = 1
STRIDE_V   = 1

DILATION_H = 1
DILATION_V = 1

OFFSET_X = 0
OFFSET_Y = 0

trials = [
    #("kernel_15_15_c_c", "c_c", ConvolveKernelAlignment.CENTER),
    #("kernel_15_15_d_l", "d_l", ConvolveKernelAlignment.BOTTOM_LEFT),
    ("kernel_15_15_d_r", "d_r", ConvolveKernelAlignment.BOTTOM_RIGHT),
    ("kernel_15_15_u_l", "u_l", ConvolveKernelAlignment.TOP_LEFT),
    ("kernel_15_15_u_r", "u_r", ConvolveKernelAlignment.TOP_RIGHT),
]

modes: List[tuple[str, ConvolvePaddingMode]] = [
    ("v", ConvolvePaddingValid()),
    ("s", ConvolvePaddingSame()),
]

OUTPUT_SUBDIR = "kernel_test_output"

ANNOTATIONS_SUBDIR = "training_labels"
IMAGES_SUBDIR = "training_images"

SOURCE_IMAGE = "proto_cells_train_118.png"
SOURCE_ANNOTATIONS = "proto_cells_train_118_annotations.json"

KERNEL_SUBDIR = "images/kernels"


def save_triplet(base: str, original, convoluted, the_mask) -> None:
    """
    Save in alphabetical order:
      base_original.png
      base_convoluted.png
      base_the_mask.png
    """
    FileUtils.save_local_bitmap(original,   OUTPUT_SUBDIR, base + "_original",   "png")
    FileUtils.save_local_bitmap(convoluted, OUTPUT_SUBDIR, base + "_convoluted", "png")
    if the_mask is not None:
        FileUtils.save_local_bitmap(the_mask, OUTPUT_SUBDIR, base + "_the_mask", "png")


for (kernel, name, alignment) in trials:
    print("kernel:", kernel, "alignment:", alignment)

    source_image = FileUtils.load_local_bitmap(IMAGES_SUBDIR, SOURCE_IMAGE)
    document = ImageAnnotationDocument.from_local_file(
        subdirectory=ANNOTATIONS_SUBDIR,
        name=SOURCE_ANNOTATIONS,
        extension=None,
    )

    mask = load_mask_white_xy_weights(KERNEL_SUBDIR, kernel)

    for (mode_tag, padding_mode) in modes:
        output_name_base = f"out_{name}_{mode_tag}"

        # ------------------------------------------------------------
        # Convolve image (must use SAME alignment as label transform if you want alignment tests to be meaningful)
        # NOTE: requires you to add kernel_alignment to Bitmap.convolve signature.
        # ------------------------------------------------------------
        bmp_conv = source_image.convolve(
            mask=mask,
            offset_x=OFFSET_X,
            offset_y=OFFSET_Y,
            stride_h=STRIDE_H,
            stride_v=STRIDE_V,
            dilation_h=DILATION_H,
            dilation_v=DILATION_V,
            padding_mode=padding_mode,
        )

        # ------------------------------------------------------------
        # Transform annotations into convolved output space
        # ------------------------------------------------------------
        doc_conv = document.transformed_with_convolve(
            mask=mask,
            offset_x=OFFSET_X,
            offset_y=OFFSET_Y,
            stride_h=STRIDE_H,
            stride_v=STRIDE_V,
            dilation_h=DILATION_H,
            dilation_v=DILATION_V,
            kernel_alignment=alignment,
            padding_mode=padding_mode,
        )

        # Render labels to bitmap in new output size
        mask_all = doc_conv.to_bitmap()

        # ------------------------------------------------------------
        # Save files in the desired alphabetical order
        # ------------------------------------------------------------
        save_triplet(output_name_base, source_image, bmp_conv, mask_all)

    print("done:", name)
