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
# Core: mask from PixelBag (fixed size, centered)
# -------------------------------------------------------------------
def mask_tensor_from_pixel_bag_fixed_size_centered(
    bag: PixelBag,
    fixed_width: int,
    fixed_height: int,
    name: str | None = None,
) -> ScorchTensor:
    """
    Create a 1 x fixed_height x fixed_width mask tensor.

    Steps:
      - Build the tight mask cropped to PixelBag.frame (same as
        mask_tensor_from_pixel_bag).
      - Place that mask centered inside a larger canvas of size
        (1, fixed_height, fixed_width).
      - The rest is zeros.

    Assumes that bag.frame width/height <= fixed_width / fixed_height.
    """
    x0, y0, w, h = bag.frame

    if w == 0 or h == 0:
        # Empty bag -> all zeros mask of requested size
        mask = np.zeros((1, fixed_height, fixed_width), dtype=np.float32)
        return ScorchTensor(mask, name=name, role="mask_fixed_size_centered")

    if w > fixed_width or h > fixed_height:
        raise ValueError(
            f"mask_tensor_from_pixel_bag_fixed_size_centered: "
            f"bag frame size ({w},{h}) exceeds fixed size "
            f"({fixed_width},{fixed_height})"
        )

    # Local tight mask
    tight = mask_tensor_from_pixel_bag(
        bag=bag,
        name=name,
    )
    tight_arr = tight.to_numpy()  # (1,h,w)

    mask = np.zeros((1, fixed_height, fixed_width), dtype=np.float32)

    # Center placement
    dx = fixed_width // 2 - w // 2
    dy = fixed_height // 2 - h // 2

    mask[:, dy:dy + h, dx:dx + w] = tight_arr

    return ScorchTensor(mask, name=name, role="mask_fixed_size_centered")


# -------------------------------------------------------------------
# Core: image + mask from PixelBag (one label, tight frame)
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
# Core: image + mask from PixelBag (fixed slab, centered)
# -------------------------------------------------------------------
def image_and_mask_from_pixel_bag_fixed_size_centered(
    bmp: Bitmap,
    bag: PixelBag,
    fixed_width: int,
    fixed_height: int,
    name: str | None = None,
    grayscale: bool = True,
) -> Tuple[ScorchTensor, ScorchTensor]:
    """
    Return (image_patch_fixed_size_centered, mask_patch_fixed_size_centered)
    for a single PixelBag.

    Steps:
      - Crop the bitmap to the PixelBag.frame (tight bounding box).
      - Verify that the frame fits inside the requested fixed size.
      - Create a new Bitmap(fixed_width, fixed_height).
      - Center-stamp the cropped bitmap into the fixed-size bitmap using:
            stamp_x = fixed_width  // 2 - w // 2
            stamp_y = fixed_height // 2 - h // 2
      - Build a matching fixed-size mask where the tight mask is
        placed at the same centered location.

    Result:
      - image_patch_fixed_size_centered: ScorchTensor, (C, fixed_height, fixed_width)
      - mask_patch_fixed_size_centered:  ScorchTensor, (1, fixed_height, fixed_width)
    """
    x0, y0, w, h = bag.frame

    if w == 0 or h == 0:
        # Empty bag -> just return all-zero tensors of the fixed size
        img_arr = np.zeros((1, fixed_height, fixed_width), dtype=np.float32)
        mask_arr = np.zeros((1, fixed_height, fixed_width), dtype=np.float32)
        img_tensor = ScorchTensor(
            img_arr,
            name=f"{name}_img_fixed_size_centered" if name else None,
            role="image_patch_fixed_size_centered",
        )
        mask_tensor = ScorchTensor(
            mask_arr,
            name=f"{name}_mask_fixed_size_centered" if name else None,
            role="mask_fixed_size_centered",
        )
        return img_tensor, mask_tensor

    if w > fixed_width or h > fixed_height:
        raise ValueError(
            f"image_and_mask_from_pixel_bag_fixed_size_centered: "
            f"bag frame size ({w},{h}) exceeds fixed size "
            f"({fixed_width},{fixed_height})"
        )

    # Defensive clamp to bitmap bounds, same as non-fixed helper
    x0 = max(0, min(x0, bmp.width  - w))
    y0 = max(0, min(y0, bmp.height - h))

    # 1) Tight crop around the bag
    cropped = bmp.crop(
        x=x0,
        y=y0,
        width=w,
        height=h,
    )

    # 2) New fixed-size bitmap and center stamp
    padded = Bitmap(fixed_width, fixed_height)

    stamp_x = fixed_width // 2 - cropped.width // 2
    stamp_y = fixed_height // 2 - cropped.height // 2

    padded.stamp(cropped, stamp_x, stamp_y)

    # 3) Convert to ScorchTensor
    img_tensor = ScorchTensor.from_bitmap(
        bmp=padded,
        name=f"{name}_img_fixed_size_centered" if name else None,
        role="image_patch_fixed_size_centered",
        grayscale=grayscale,
    )

    # 4) Matching fixed-size mask
    mask_tensor = mask_tensor_from_pixel_bag_fixed_size_centered(
        bag=bag,
        fixed_width=fixed_width,
        fixed_height=fixed_height,
        name=f"{name}_mask_fixed_size_centered" if name else None,
    )

    # Final sanity check: shapes must match
    _, h_m, w_m = mask_tensor.shape
    _, h_i, w_i = img_tensor.shape
    if (h_m, w_m) != (h_i, w_i):
        raise ValueError(
            f"Fixed-size mask/image size mismatch: "
            f"mask=({h_m},{w_m}), image=({h_i},{w_i})"
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

    Uses the tight PixelBag.frame (no padding).
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


# -------------------------------------------------------------------
# High-level: iterate fixed-size centered patches over a pair
# -------------------------------------------------------------------
def iter_label_patches_from_pair_fixed_size_centered(
    pair: AnnotationImagePair,
    fixed_width: int,
    fixed_height: int,
    grayscale: bool = True,
):
    """
    Yield (image_patch_fixed_size_centered, mask_patch_fixed_size_centered,
    label_name) for every label in a single AnnotationImagePair.

    Each patch is exactly (fixed_height, fixed_width) with the original
    PixelBag.frame content centered in the slab.
    """
    bmp = Bitmap.with_image(pair.image_path)
    doc = pair.document

    for label in doc.data.labels:
        bag = label.pixel_bag
        img_patch, mask_patch = image_and_mask_from_pixel_bag_fixed_size_centered(
            bmp=bmp,
            bag=bag,
            fixed_width=fixed_width,
            fixed_height=fixed_height,
            name=label.name,
            grayscale=grayscale,
        )
        yield img_patch, mask_patch, label.name
