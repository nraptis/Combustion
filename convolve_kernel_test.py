from __future__ import annotations

from image_tools.mask_loader import load_mask_white_xy_weights

from filesystem.file_utils import FileUtils
from filesystem.file_io import FileIO

from typing import List, Optional, Tuple

from image.bitmap import Bitmap
from image.rgba import RGBA

from labels.pixel_bag import PixelBag
from labels.data_label import DataLabel
from labels.image_annotation_document import ImageAnnotationDocument

from image.convolve_kernel_alignment import ConvolveKernelAlignment
from image.convolve_padding_mode import (
    ConvolvePaddingSame,
    ConvolvePaddingValid,
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
    ConvolvePaddingMode,
)

import numpy as np
import torch
import torch.nn.functional as F

OUTPUT_SUBDIR = "kernel_test_output"

ANNOTATIONS_SUBDIR = "training_labels"
IMAGES_SUBDIR = "training_images"

SOURCE_IMAGE = "proto_cells_train_118.png"
SOURCE_ANNOTATIONS = 'proto_cells_train_118_annotations.json'

KERNEL_SUBDIR = "images/kernels"

"""
kernel_1_1
kernel_1_2
kernel_2_1
kernel_2_2
kernel_2_3
kernel_3_2
kernel_3_3_c_c
kernel_3_3_d_l
kernel_3_3_d_r
kernel_3_3_u_l
kernel_3_3_u_r
kernel_3_3
kernel_corner_3_8_d_l
kernel_corner_3_8_d_r
kernel_corner_3_8_u_l
kernel_corner_3_8_u_r
kernel_corner_4_5_d_l
kernel_corner_4_5_d_r
kernel_corner_4_5_u_l
kernel_corner_4_5_u_r
kernel_corner_5_4_d_l
kernel_corner_5_4_d_r
kernel_corner_5_4_u_l
kernel_corner_5_4_u_r
kernel_corner_8_3_d_l
kernel_corner_8_3_d_r
kernel_corner_8_3_u_l
kernel_corner_8_3_u_r
"""

trials = [
    ("kernel_15_15_c_c", "ctr", ConvolveKernelAlignment.CENTER),
    ("kernel_15_15_d_l", "dl", ConvolveKernelAlignment.BOTTOM_LEFT),
]

for (kernel, name, alignment) in trials:

    output_name_base = "out_" + name
    
    # the original bitmap
    output_name_original = output_name_base + "_original"

    #convolve using bitmap
    output_name_internal = output_name_base + "_int"

    #convolve using torch (one liner here + helper mthod above)
    output_name_torch = output_name_base + "_int"


    output_name_mask_one = output_name_base + "_mask_one"
    output_name_mask_all = output_name_base + "_mask_all"

    #output name example 

    print("alignment is ", alignment)

    source_image = FileUtils.load_local_bitmap(IMAGES_SUBDIR, SOURCE_IMAGE)
    print(f"image=>{source_image.width}x{source_image.height}")
    
    document = ImageAnnotationDocument.from_local_file(
                    subdirectory=ANNOTATIONS_SUBDIR,
                    name=SOURCE_ANNOTATIONS,
                    extension=None,
                )
    
    print(document)

    for label in document.data.labels:
        pixel_bag = label.pixel_bag
        print(pixel_bag)

    mask = load_mask_white_xy_weights(KERNEL_SUBDIR, kernel)
    print(mask)

    # save bitmap with one-liner
    FileUtils.save_local_bitmap(source_image, OUTPUT_SUBDIR, output_name_original, "png")

