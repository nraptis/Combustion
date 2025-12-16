from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from image.bitmap import Bitmap
from image.rgba import RGBA
from image.convolve_padding_mode import (
    ConvolvePaddingValid,
    ConvolvePaddingSame,
    ConvolvePaddingOffsetSame,
    ConvolvePaddingOffsetValid,
    ConvolvePaddingMode,
)

# ------------------------------------------------------------
# Build a 4x4 "coordinate bitmap"
# Each pixel encodes (x,y) into color so mismatches are obvious.
# ------------------------------------------------------------
def make_coord_bitmap_4x4() -> Bitmap:
    bmp = Bitmap(4, 4)
    for x in range(4):
        for y in range(4):
            px = bmp.rgba[x][y]
            # Unique-ish but small: R encodes x, G encodes y, B encodes x+y
            px.ri = 10 + x
            px.gi = 20 + y
            px.bi = 30 + (x + y)
            px.ai = 255
    return bmp


# ------------------------------------------------------------
# One-hot mask: mask[x][y] with 1.0 at (kx,ky)
# ------------------------------------------------------------
def one_hot_mask(kw: int, kh: int, kx: int, ky: int) -> List[List[float]]:
    mask: List[List[float]] = []
    for x in range(kw):
        col = []
        for y in range(kh):
            col.append(1.0 if (x == kx and y == ky) else 0.0)
        mask.append(col)
    return mask


# ------------------------------------------------------------
# SAME padding split (must match your Bitmap.convolve() logic)
# (Non-offset SAME only; for offset SAME we still use SAME formula but with k_budget)
# ------------------------------------------------------------
def ceil_div(n: int, d: int) -> int:
    return n // d + (1 if (n % d) != 0 else 0)

def same_pad_split(W: int, H: int, k_budget_w: int, k_budget_h: int, sh: int, sv: int) -> Tuple[int,int,int,int]:
    out_w = ceil_div(W, sh)
    out_h = ceil_div(H, sv)

    pad_total_w = max(0, (out_w - 1) * sh + k_budget_w - W)
    pad_total_h = max(0, (out_h - 1) * sv + k_budget_h - H)

    pad_left = pad_total_w // 2
    pad_right = pad_total_w - pad_left
    pad_top = pad_total_h // 2
    pad_bottom = pad_total_h - pad_top
    return pad_left, pad_right, pad_top, pad_bottom


# ------------------------------------------------------------
# Expected output for one-hot sampling under your TOP_LEFT start contract
# We compute expected as BGRA uint8 (because export_opencv is BGRA)
# ------------------------------------------------------------
def expected_one_hot(
    src_bmp: Bitmap,
    kw: int, kh: int,
    kx: int, ky: int,
    padding_mode: ConvolvePaddingMode,
    offset_x: int, offset_y: int,
    sh: int = 1, sv: int = 1,
    dh: int = 1, dv: int = 1,
) -> np.ndarray:
    src = src_bmp.export_opencv()  # (H,W,4) BGRA uint8
    H, W = src.shape[0], src.shape[1]

    # effective footprint (for dilation; here usually 1)
    k_eff_w = (kw - 1) * dh + 1
    k_eff_h = (kh - 1) * dv + 1

    # offset mode expands "budget"
    max_ox = 0
    max_oy = 0
    is_offset_mode = isinstance(padding_mode, (ConvolvePaddingOffsetSame, ConvolvePaddingOffsetValid))
    if is_offset_mode:
        max_ox = int(padding_mode.max_offset_x)
        max_oy = int(padding_mode.max_offset_y)

    k_budget_w = k_eff_w + 2 * max_ox
    k_budget_h = k_eff_h + 2 * max_oy

    # output size + pad
    if isinstance(padding_mode, (ConvolvePaddingSame, ConvolvePaddingOffsetSame)):
        pad_left, pad_right, pad_top, pad_bottom = same_pad_split(W, H, k_budget_w, k_budget_h, sh, sv)
        out_w = ceil_div(W, sh)
        out_h = ceil_div(H, sv)
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0
        # VALID / OFFSET_VALID use budget for output size
        out_w = (W - k_budget_w) // sh + 1
        out_h = (H - k_budget_h) // sv + 1
        if out_w <= 0 or out_h <= 0:
            return np.zeros((0,0,4), dtype=np.uint8)

    # build expected output
    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    # NOTE: your Bitmap.convolve start_x0/start_y0 is:
    # - base_x/base_y = max_offset when offset modes, else 0
    base_x = max_ox if is_offset_mode else 0
    base_y = max_oy if is_offset_mode else 0

    start_x0 = base_x + int(offset_x)
    start_y0 = base_y + int(offset_y)

    # one-hot sample position inside kernel footprint
    # source coord for output (ox,oy) in *padded* coordinates:
    # px = start_x0 + (kx*dh) + ox*sh
    # py = start_y0 + (ky*dv) + oy*sv
    for oy in range(out_h):
        for ox in range(out_w):
            px = start_x0 + (kx * dh) + ox * sh
            py = start_y0 + (ky * dv) + oy * sv

            # convert padded coord -> source coord for SAME cases
            sx = px - pad_left
            sy = py - pad_top

            if 0 <= sx < W and 0 <= sy < H:
                out[oy, ox, :] = src[sy, sx, :]
            else:
                out[oy, ox, :] = np.array([0, 0, 0, 0], dtype=np.uint8)

    return out


def assert_bgra_equal(a: np.ndarray, b: np.ndarray, label: str) -> None:
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape mismatch {a.shape} vs {b.shape}")
    if not np.array_equal(a, b):
        # locate first mismatch for a friendly error
        ys, xs, cs = np.where(a != b)
        y0, x0, c0 = int(ys[0]), int(xs[0]), int(cs[0])
        raise AssertionError(
            f"{label}: first mismatch at (x={x0}, y={y0}, c={c0}) "
            f"a={a[y0,x0,:].tolist()} b={b[y0,x0,:].tolist()}"
        )


def run_case(
    *,
    padding_mode: ConvolvePaddingMode,
    offset_x: int,
    offset_y: int,
    kw: int, kh: int,
    kx: int, ky: int,
    sh: int = 1, sv: int = 1,
    dh: int = 1, dv: int = 1,
) -> None:
    src = make_coord_bitmap_4x4()
    mask = one_hot_mask(kw, kh, kx, ky)

    out_bmp = src.convolve(
        mask=mask,
        offset_x=offset_x,
        offset_y=offset_y,
        stride_h=sh,
        stride_v=sv,
        dilation_h=dh,
        dilation_v=dv,
        padding_mode=padding_mode,
    )
    got = out_bmp.export_opencv()

    exp = expected_one_hot(
        src_bmp=src,
        kw=kw, kh=kh,
        kx=kx, ky=ky,
        padding_mode=padding_mode,
        offset_x=offset_x, offset_y=offset_y,
        sh=sh, sv=sv,
        dh=dh, dv=dv,
    )

    assert_bgra_equal(got, exp, label=f"{padding_mode.__class__.__name__} off=({offset_x},{offset_y}) hot=({kx},{ky})")


def main() -> None:
    kw = kh = 3

    # “Which pixel do we sample from the 3x3 window?”
    HOTS = {
        "TL": (0, 0),
        "TR": (2, 0),
        "C":  (1, 1),
        "BL": (0, 2),
        "BR": (2, 2),
    }

    OFFSETS = [
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    modes: List[ConvolvePaddingMode] = [
        ConvolvePaddingValid(),
        ConvolvePaddingSame(),
        ConvolvePaddingOffsetValid(max_offset_x=2, max_offset_y=2),
        ConvolvePaddingOffsetSame(max_offset_x=2, max_offset_y=2),
    ]

    for mode in modes:
        for (ox, oy) in OFFSETS:
            for name, (kx, ky) in HOTS.items():
                run_case(
                    padding_mode=mode,
                    offset_x=ox,
                    offset_y=oy,
                    kw=kw, kh=kh,
                    kx=kx, ky=ky,
                    sh=1, sv=1,
                    dh=1, dv=1,
                )
                print("PASS", mode.__class__.__name__, "off", (ox,oy), "hot", name)

    print("\nAll one-hot contract tests passed ✅")


if __name__ == "__main__":
    main()
