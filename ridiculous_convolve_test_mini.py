from __future__ import annotations

import os
import random
from typing import List

from image.bitmap import Bitmap
from filesystem.file_utils import FileUtils
from filesystem.file_io import FileIO

from image.convolve_padding_mode import ConvolvePaddingSame, ConvolvePaddingValid

import numpy as np
import torch
import torch.nn.functional as F


def normalize_mask_l1(mask: List[List[float]]) -> List[List[float]]:
    kw = len(mask)
    kh = len(mask[0]) if kw > 0 else 0
    s = 0.0
    for x in range(kw):
        for y in range(kh):
            s += abs(mask[x][y])
    if s <= 1e-12:
        return mask
    inv = 1.0 / s
    for x in range(kw):
        for y in range(kh):
            mask[x][y] *= inv
    return mask

# ----------------------------
# same kernel builder as before
# ----------------------------
def make_kernel_mask(kw: int, kh: int) -> List[List[float]]:
    kw = int(kw)
    kh = int(kh)
    cx = (kw - 1) * 0.5
    cy = (kh - 1) * 0.5
    sx = max(1.0, kw * 0.35)
    sy = max(1.0, kh * 0.35)

    mask: List[List[float]] = [[0.0 for _y in range(kh)] for _x in range(kw)]
    for x in range(kw):
        for y in range(kh):
            dx = (x - cx) / sx
            dy = (y - cy) / sy
            g = float(np.exp(-(dx * dx + dy * dy)))
            ripple = 1.0 if ((x + y) & 1) == 0 else -0.25
            mask[x][y] = g * ripple
    return mask


def convolve_torch_reference(
    bmp: Bitmap,
    mask: List[List[float]],
    stride_h: int,
    stride_v: int,
    dilation_h: int,
    dilation_v: int,
    padding_mode,
    device: str = "cpu",
) -> Bitmap:
    kw = len(mask)
    kh = len(mask[0]) if kw > 0 else 0

    mask_np = np.empty((kh, kw), dtype=np.float32)
    for x in range(kw):
        for y in range(kh):
            mask_np[y, x] = float(mask[x][y])

    src_bgra_u8 = bmp.export_opencv()  # (H,W,4)
    x = torch.from_numpy(src_bgra_u8).to(device=device)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous().float()  # [1,4,H,W]

    k = torch.from_numpy(mask_np).to(device=device).float()
    w = k.view(1, 1, kh, kw).repeat(4, 1, 1, 1).contiguous()

    sh = int(stride_h)
    sv = int(stride_v)
    dh = int(dilation_h)
    dv = int(dilation_v)

    if isinstance(padding_mode, ConvolvePaddingSame):
        H = int(src_bgra_u8.shape[0])
        W = int(src_bgra_u8.shape[1])

        k_eff_w = (kw - 1) * dh + 1
        k_eff_h = (kh - 1) * dv + 1

        out_w = W // sh + (1 if (W % sh) != 0 else 0)
        out_h = H // sv + (1 if (H % sv) != 0 else 0)

        pad_total_w = max(0, (out_w - 1) * sh + k_eff_w - W)
        pad_total_h = max(0, (out_h - 1) * sv + k_eff_h - H)

        pad_left = pad_total_w // 2
        pad_right = pad_total_w - pad_left
        pad_top = pad_total_h // 2
        pad_bottom = pad_total_h - pad_top

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        out = F.conv2d(
            x, w, bias=None,
            stride=(sv, sh),
            padding=0,
            dilation=(dv, dh),
            groups=4,
        )
    else:
        out = F.conv2d(
            x, w, bias=None,
            stride=(sv, sh),
            padding=0,
            dilation=(dv, dh),
            groups=4,
        )

    out_u8 = torch.clamp(out + 0.5, 0.0, 255.0).to(torch.uint8)
    out_bgra_u8 = (
        out_u8.squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )

    result = Bitmap(int(out_bgra_u8.shape[1]), int(out_bgra_u8.shape[0]))
    result.import_opencv(out_bgra_u8)
    return result


def quick_visual_one_case():
    random.seed(123)

    # ------------------------------------------------------------
    # Load base image Im013_1 (1712 x 1368)
    # ------------------------------------------------------------
    base = Bitmap.with_local_image(subdirectory="input", name="Im013_1", extension=None)

    # ------------------------------------------------------------
    # Make a 100x100 and stamp base with negative random offsets
    # ------------------------------------------------------------
    bmp = Bitmap(100, 100)
    sx = -random.randint(0, 400)
    sy = -random.randint(0, 400)
    bmp.stamp(base, sx, sy)

    # ------------------------------------------------------------
    # Choose ONE kernel + params (adjust if you want)
    # ------------------------------------------------------------
    kw, kh = 11, 11
    mask = make_kernel_mask(kw, kh)
    mask = normalize_mask_l1(mask)

    stride_h, stride_v = 1, 1
    dilation_h, dilation_v = 1, 1

    # Try SAME first; VALID often whiffs with big kernels on small images.
    padding_mode = ConvolvePaddingSame()

    # ------------------------------------------------------------
    # Run Bitmap + Torch
    # ------------------------------------------------------------
    out_bitmap = bmp.convolve(
        mask=mask,
        offset_x=0,
        offset_y=0,
        stride_h=stride_h,
        stride_v=stride_v,
        dilation_h=dilation_h,
        dilation_v=dilation_v,
        padding_mode=padding_mode,
    )

    try:
        out_torch = convolve_torch_reference(
            bmp,
            mask,
            stride_h=stride_h,
            stride_v=stride_v,
            dilation_h=dilation_h,
            dilation_v=dilation_v,
            padding_mode=padding_mode,
            device="cpu",
        )
    except Exception as e:
        print(f"❌ Torch threw: {type(e).__name__}: {e}")
        out_torch = Bitmap()  # empty

    # ------------------------------------------------------------
    # Save images
    # ------------------------------------------------------------
    #out_dir = FileIO.local_file(subdirectory="convolve", name="", extension=None)
    #os.makedirs(out_dir, exist_ok=True)

    FileUtils.save_local_bitmap(bmp,        subdirectory="convolve", name="original.png",          extension=None)
    FileUtils.save_local_bitmap(out_bitmap, subdirectory="convolve", name="convolved_bitmap.png",  extension=None)
    FileUtils.save_local_bitmap(out_torch,  subdirectory="convolve", name="convolved_torch.png",   extension=None)

    print("✅ Saved:")
    print("   convolve/original.png")
    print("   convolve/convolved_bitmap.png")
    print("   convolve/convolved_torch.png")
    print(f"   stamp_offset=({sx},{sy}) kernel={kw}x{kh} stride=({stride_h},{stride_v}) dil=({dilation_h},{dilation_v}) mode=SAME")
    print(f"   out_bitmap={out_bitmap.width}x{out_bitmap.height} out_torch={out_torch.width}x{out_torch.height}")


quick_visual_one_case()