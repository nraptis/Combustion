# scorch/tensor_load_helpers.py

from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np

from labels.pixel_bag import PixelBag
from scorch.scorch_tensor import ScorchTensor
from image.bitmap import Bitmap
from scorch.data_loader import AnnotationImagePair, DataLoader


# -------------------------------------------------------------------
# Core: mask from PixelBag (cropped to PixelBag.frame)
# -------------------------------------------------------------------
def mask_tensor_from_pixel_bag(
    bag: PixelBag,
    name: str | None = None,
) -> ScorchTensor:
    """
    Create a 1 x H x W mask tensor cropped to the PixelBag's frame.
    Pixels in the bag -> 1.0, everything else -> 0.0.
    """
    x0, y0, w, h = bag.frame

    if w == 0 or h == 0:
        return ScorchTensor(
            np.zeros((0,), dtype=np.float32),
            name=name,
            role="mask",
        )

    mask = np.zeros((1, h, w), dtype=np.float32)

    for x, y in bag:
        mx = x - x0  # local coords inside bbox
        my = y - y0
        if 0 <= mx < w and 0 <= my < h:
            mask[0, my, mx] = 1.0

    return ScorchTensor(mask, name=name, role="mask")


# -------------------------------------------------------------------
# Core: image + mask from PixelBag (one label)
# -------------------------------------------------------------------
def image_and_mask_from_pixel_bag(
    bmp: Bitmap,
    bag: PixelBag,
    name: str | None = None,
    grayscale: bool = True,
) -> Tuple[ScorchTensor, ScorchTensor]:
    """
    Return (image_patch, mask_patch) for a single PixelBag.

    Both tensors are cropped to PixelBag.frame and share the same H, W.
    """
    x0, y0, w, h = bag.frame

    if w == 0 or h == 0:
        empty = ScorchTensor(
            np.zeros((0,), dtype=np.float32),
            name=name,
            role="empty",
        )
        return empty, empty

    # Defensive clamp to bitmap bounds
    x0 = max(0, min(x0, bmp.width  - w))
    y0 = max(0, min(y0, bmp.height - h))

    img_tensor = ScorchTensor.from_bitmap_crop(
        bmp=bmp,
        x=x0,
        y=y0,
        width=w,
        height=h,
        name=f"{name}_img" if name else None,
        role="image_patch",
        grayscale=grayscale,
    )

    mask_tensor = mask_tensor_from_pixel_bag(
        bag=bag,
        name=f"{name}_mask" if name else None,
    )

    # Simple shape sanity check
    _, h_m, w_m = mask_tensor.shape
    _, h_i, w_i = img_tensor.shape
    if (h_m, w_m) != (h_i, w_i):
        raise ValueError(
            f"Mask/image size mismatch: mask=({h_m},{w_m}), image=({h_i},{w_i})"
        )

    return img_tensor, mask_tensor


# -------------------------------------------------------------------
# High-level: iterate (image_patch, mask_patch, label_name) over a pair
# -------------------------------------------------------------------
def iter_label_patches_from_pair(
    pair: AnnotationImagePair,
    grayscale: bool = True,
):
    """
    Yield (image_patch, mask_patch, label_name) for every label
    in a single AnnotationImagePair.
    """
    bmp = Bitmap.with_image(pair.image_path)
    doc = pair.document

    # Adjust attribute names if your document layout differs
    for label in doc.data.labels:
        bag = label.pixel_bag
        img_patch, mask_patch = image_and_mask_from_pixel_bag(
            bmp=bmp,
            bag=bag,
            name=label.name,
            grayscale=grayscale,
        )
        yield img_patch, mask_patch, label.name
        