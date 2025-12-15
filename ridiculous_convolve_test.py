# ridiculous_convolve_test.py

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from image.bitmap import Bitmap
from image.convolve_padding_mode import (
    ConvolvePaddingSame,
    ConvolvePaddingValid,
)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
IMAGE_NAME = "Im013_1"
IMAGE_SUBDIR = "input"          # your FileUtils local scheme
TOLERANCE = 2

MAX_BMP_W = 64
MAX_BMP_H = 64

MAX_KERNEL_W = 5
MAX_KERNEL_H = 5

MAX_STRIDE = 2
MAX_DILATION = 2

SEED = 911911


def normalize_mask_l1(mask: List[List[float]]) -> List[List[float]]:
    s = 0.0
    for x in range(len(mask)):
        for y in range(len(mask[0])):
            s += abs(mask[x][y])
    if s > 0:
        inv = 1.0 / s
        for x in range(len(mask)):
            for y in range(len(mask[0])):
                mask[x][y] *= inv
    return mask


# ------------------------------------------------------------
# Kernel builder (deterministic, non-trivial)
# mask[x][y] layout (your weird mask format)
# ------------------------------------------------------------
def make_kernel_mask(kw: int, kh: int) -> List[List[float]]:
    kw = int(kw)
    kh = int(kh)
    if kw <= 0 or kh <= 0:
        return []

    # A simple, stable pattern: center-weighted with mild ringing.
    # Deterministic and non-zero-ish, so we actually exercise math.
    cx = (kw - 1) * 0.5
    cy = (kh - 1) * 0.5
    sx = max(1.0, kw * 0.35)
    sy = max(1.0, kh * 0.35)

    mask: List[List[float]] = [[0.0 for _y in range(kh)] for _x in range(kw)]
    for x in range(kw):
        for y in range(kh):
            dx = (x - cx) / sx
            dy = (y - cy) / sy
            g = np.exp(-(dx * dx + dy * dy))  # gaussian-ish
            # add a mild alternating sign ripple to stress edge cases
            ripple = 1.0 if ((x + y) & 1) == 0 else -0.25
            mask[x][y] = float(g * ripple)

    return mask


# ------------------------------------------------------------
# Torch reference (depthwise conv2d) that matches your Bitmap path:
# - Source is BGRA u8 -> float
# - SAME/VALID semantics
# - stride/dilation supported
# - offset forced 0 (as requested)
# - output: BGRA u8 with +0.5 rounding then clamp
# ------------------------------------------------------------
def convolve_torch_reference(
    bmp: Bitmap,
    mask: List[List[float]],
    stride_h: int,
    stride_v: int,
    dilation_h: int,
    dilation_v: int,
    padding_mode,  # ConvolvePaddingSame() or ConvolvePaddingValid()
    device: str = "cpu",
) -> Bitmap:
    kw = len(mask)
    kh = len(mask[0]) if kw > 0 else 0

    # mask[x][y] -> mask_np[ky,kx]
    mask_np = np.empty((kh, kw), dtype=np.float32)
    for x in range(kw):
        col = mask[x]
        for y in range(kh):
            mask_np[y, x] = float(col[y])

    # Source BGRA u8 -> torch [1,4,H,W] float
    src_bgra_u8 = bmp.export_opencv()  # (H,W,4) uint8 BGRA
    x = torch.from_numpy(src_bgra_u8).to(device=device)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous().float()  # [1,4,H,W]

    # Depthwise weights [4,1,kh,kw]
    k = torch.from_numpy(mask_np).to(device=device).float()
    w = k.view(1, 1, kh, kw).repeat(4, 1, 1, 1).contiguous()

    sh = int(stride_h)
    sv = int(stride_v)
    dh = int(dilation_h)
    dv = int(dilation_v)

    if sh <= 0 or sv <= 0 or dh <= 0 or dv <= 0:
        raise ValueError("invalid stride/dilation")

    # SAME/VALID padding:
    # For SAME we do explicit F.pad on x then VALID conv (padding=0).
    # This matches the way your Bitmap does pad-then-sample.
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

        # F.pad expects (left,right,top,bottom)
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        out = F.conv2d(
            x, w, bias=None,
            stride=(sv, sh),
            padding=0,
            dilation=(dv, dh),
            groups=4,
        )

    else:
        # VALID
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


# ------------------------------------------------------------
# Test driver
# ------------------------------------------------------------
def main():
    random.seed(SEED)

    base = Bitmap.with_local_image(subdirectory=IMAGE_SUBDIR, name=IMAGE_NAME, extension=None)
    if base.width <= 0 or base.height <= 0:
        raise RuntimeError(f"Failed to load base image: {IMAGE_SUBDIR}/{IMAGE_NAME}")

    padding_modes = [ConvolvePaddingSame(), ConvolvePaddingValid()]

    total = 0
    ok = 0
    fail = 0
    torch_exceptions = 0
    bitmap_whiffs = 0
    mismatches = 0

    # Outer loop: all bmp sizes (print each height change)
    for bmp_h in range(1, MAX_BMP_H + 1):
        print(f"\n=== bmp_height={bmp_h} ===")
        height_ok = 0
        height_fail = 0

        for bmp_w in range(1, MAX_BMP_W + 1):

            # Build test bitmap and stamp base at negative random offsets
            test_bmp = Bitmap(bmp_w, bmp_h)
            x0 = -random.randint(0, 400)
            y0 = -random.randint(0, 400)
            test_bmp.stamp(base, x0, y0)

            for kw in range(1, MAX_KERNEL_W + 1):
                for kh in range(1, MAX_KERNEL_H + 1):
                    mask = make_kernel_mask(kw, kh)
                    mask = normalize_mask_l1(mask)

                    for sh in range(1, MAX_STRIDE + 1):
                        for sv in range(1, MAX_STRIDE + 1):
                            for dh in range(1, MAX_DILATION + 1):
                                for dv in range(1, MAX_DILATION + 1):
                                    for pm in padding_modes:
                                        total += 1

                                        # Bitmap convolve (offset is always 0)
                                        try:
                                            out_bmp = test_bmp.convolve(
                                                mask=mask,
                                                offset_x=0,
                                                offset_y=0,
                                                stride_h=sh,
                                                stride_v=sv,
                                                dilation_h=dh,
                                                dilation_v=dv,
                                                padding_mode=pm,
                                            )
                                        except Exception as e:
                                            # Bitmap should NOT throw in this new regime; count it as fail
                                            fail += 1
                                            height_fail += 1
                                            print(
                                                "❌ Bitmap exception "
                                                f"bmp={bmp_w}x{bmp_h} k={kw}x{kh} "
                                                f"stride=({sh},{sv}) dil=({dh},{dv}) pm={Bitmap.padding_mode_string(pm)} "
                                                f"exc={type(e).__name__}: {e}"
                                            )
                                            continue

                                        # Torch reference
                                        torch_ok = True
                                        torch_out = None
                                        torch_exc = None
                                        try:
                                            torch_out = convolve_torch_reference(
                                                test_bmp, mask, sh, sv, dh, dv, pm, device="cpu"
                                            )
                                        except Exception as e:
                                            torch_ok = False
                                            torch_exc = e

                                        # If torch threw, Bitmap must whiff (empty) for SAME or VALID too
                                        if not torch_ok:
                                            torch_exceptions += 1
                                            # Bitmap should return empty bitmap on whiff
                                            if out_bmp.width != 0 or out_bmp.height != 0:
                                                mismatches += 1
                                                fail += 1
                                                height_fail += 1
                                                print(
                                                    "❌ Torch threw but Bitmap did NOT whiff "
                                                    f"bmp={bmp_w}x{bmp_h} k={kw}x{kh} "
                                                    f"stride=({sh},{sv}) dil=({dh},{dv}) pm={Bitmap.padding_mode_string(pm)} "
                                                    f"torch_exc={type(torch_exc).__name__}: {torch_exc} "
                                                    f"bitmap_out={out_bmp.width}x{out_bmp.height}"
                                                )
                                            else:
                                                bitmap_whiffs += 1
                                                ok += 1
                                                height_ok += 1
                                            continue

                                        # Torch succeeded: Bitmap must have a non-empty output and match
                                        if out_bmp.width == 0 or out_bmp.height == 0:
                                            mismatches += 1
                                            fail += 1
                                            height_fail += 1
                                            print(
                                                "❌ Bitmap whiffed but Torch succeeded "
                                                f"bmp={bmp_w}x{bmp_h} k={kw}x{kh} "
                                                f"stride=({sh},{sv}) dil=({dh},{dv}) pm={Bitmap.padding_mode_string(pm)} "
                                                f"torch_out={torch_out.width}x{torch_out.height}"
                                            )
                                            continue

                                        if not out_bmp.compare(torch_out, tolerance=TOLERANCE):
                                            mismatches += 1
                                            fail += 1
                                            height_fail += 1
                                            print(
                                                "❌ Output mismatch "
                                                f"bmp={bmp_w}x{bmp_h} k={kw}x{kh} "
                                                f"stride=({sh},{sv}) dil=({dh},{dv}) pm={Bitmap.padding_mode_string(pm)} "
                                                f"out_bmp={out_bmp.width}x{out_bmp.height} torch={torch_out.width}x{torch_out.height}"
                                            )
                                            # Show first diff detail if you want it:
                                            # print("   first_diff:", first_diff(out_bmp, torch_out))
                                            continue

                                        ok += 1
                                        height_ok += 1

            # small periodic progress ping per width
            if (bmp_w % 25) == 0:
                print(
                    f"  bmp_width={bmp_w} "
                    f"height_ok={height_ok} height_fail={height_fail} "
                    f"total_ok={ok} total_fail={fail} total={total}"
                )

        print(
            f"=== SUMMARY height={bmp_h}: ok={height_ok} fail={height_fail} "
            f"(running totals: ok={ok} fail={fail} torch_ex={torch_exceptions} "
            f"bitmap_whiffs={bitmap_whiffs} mismatches={mismatches} total={total}) ==="
        )


if __name__ == "__main__":
    main()
