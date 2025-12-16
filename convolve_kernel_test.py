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


ANNOTATIONS_SUBDIR = "training_labels"
IMAGES_SUBDIR = "training_images"

SOURCE_IMAGE = "proto_cells_train_118.png"
SOURCE_ANNOTATIONS = 'proto_cells_train_118_annotations.json'

KERNEL_SUBDIR = "images/kernels"
kernels = ["kernel_corner_4_5_d_l"]

for kernel in kernels:

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


