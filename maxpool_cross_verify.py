# scorch/maxpool_cross_verify.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scorch.scorch_max_pool2d import ScorchMaxPool2d
from scorch.scorch_max_pool2d_fast import ScorchMaxPool2dFast


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

NUM_TRIALS = 1024


# ------------------------------------------------------------
# Utility: tolerant equality check
# ------------------------------------------------------------

def assert_allclose(a, b, atol=1e-4, rtol=1e-4, name=""):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.abs(a - b)
        max_diff = float(diff.max())
        raise AssertionError(
            f"[{name}] max |a-b| = {max_diff}, "
            f"atol={atol}, rtol={rtol}"
        )


# ------------------------------------------------------------
# Random param + input generators
# ------------------------------------------------------------

def random_int(low: int, high: int) -> int:
    """
    Inclusive integer RNG: [low, high].
    """
    return int(np.random.randint(low, high + 1))


def make_random_pool_params():
    """
    Random (but valid) 2D max-pooling hyperparameters.

    Constraints:
      - stride, dilation in [1..4]
      - padding in [0..4] but ALSO:
            pad_h <= floor(k_h / 2)
            pad_w <= floor(k_w / 2)
        to satisfy PyTorch's max-pool padding rule.

      - Output size must be >= 1x1.
    """
    while True:
        C = random_int(1, 4)

        H = random_int(5, 20)
        W = random_int(5, 20)

        # Kernel size 1..5
        k_h = random_int(1, 5)
        k_w = random_int(1, 5)

        # stride, dilation ~ 1..4 (can't be 0)
        stride_h = random_int(1, 4)
        stride_w = random_int(1, 4)

        d_h = random_int(1, 4)
        d_w = random_int(1, 4)

        # PyTorch requirement: pad <= floor(kernel / 2)
        max_pad_h = min(4, k_h // 2)
        max_pad_w = min(4, k_w // 2)

        pad_h = random_int(0, max_pad_h)
        pad_w = random_int(0, max_pad_w)

        batch = random_int(1, 4)

        # 1) Kernel window must fit in padded input
        #    (a bit redundant now, but keeps us extra safe)
        if k_h > H + 2 * pad_h or k_w > W + 2 * pad_w:
            continue

        # 2) Output size must be >= 1x1 (PyTorch requirement)
        H_out = ((H + 2 * pad_h - d_h * (k_h - 1) - 1) // stride_h) + 1
        W_out = ((W + 2 * pad_w - d_w * (k_w - 1) - 1) // stride_w) + 1

        if H_out <= 0 or W_out <= 0:
            continue

        # Valid configuration
        return {
            "N": batch,
            "C": C,
            "H": H,
            "W": W,
            "kernel": (k_h, k_w),
            "stride": (stride_h, stride_w),
            "padding": (pad_h, pad_w),
            "dilation": (d_h, d_w),
            "H_out": H_out,
            "W_out": W_out,
        }
    
def make_random_input(N, C, H, W):
    """
    Random float32 input in a reasonable range.
    """
    return np.random.uniform(-3, 3, size=(N, C, H, W)).astype(np.float32)


# ------------------------------------------------------------
# Core test
# ------------------------------------------------------------

def cross_verify_once(trial_index: int):
    p = make_random_pool_params()

    N      = p["N"]
    C      = p["C"]
    H      = p["H"]
    W      = p["W"]
    ksize  = p["kernel"]
    stride = p["stride"]
    pad    = p["padding"]
    dial   = p["dilation"]

    # --------------------------------------------------------
    # Create Scorch layers
    # --------------------------------------------------------
    pool_slow = ScorchMaxPool2d(
        kernel_size=ksize,
        stride=stride,
        padding=pad,
        dilation=dial,
    )

    pool_fast = ScorchMaxPool2dFast(
        kernel_size=ksize,
        stride=stride,
        padding=pad,
        dilation=dial,
    )

    # --------------------------------------------------------
    # Create PyTorch reference pool
    # --------------------------------------------------------
    pool_torch = nn.MaxPool2d(
        kernel_size=ksize,
        stride=stride,
        padding=pad,
        dilation=dial,
        return_indices=False,
        ceil_mode=False,
    )

    # --------------------------------------------------------
    # Random input
    # --------------------------------------------------------
    x = make_random_input(N, C, H, W)
    x_torch = torch.from_numpy(x.copy()).requires_grad_(True)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    y_slow = pool_slow.forward(x.copy())
    y_fast = pool_fast.forward(x.copy())
    y_torch = pool_torch(x_torch)
    y_np = y_torch.detach().cpu().numpy()

    assert_allclose(y_slow, y_fast, name="slow vs fast forward")
    assert_allclose(y_fast, y_np, name="fast vs torch forward")
    assert_allclose(y_slow, y_np,   name="slow vs torch forward")

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------
    grad_out = np.random.randn(*y_slow.shape).astype(np.float32)
    grad_out_torch = torch.from_numpy(grad_out.copy())

    gx_slow = pool_slow.backward(grad_out.copy())
    gx_fast = pool_fast.backward(grad_out.copy())

    # Torch backward
    pool_torch.zero_grad()
    if x_torch.grad is not None:
        x_torch.grad.zero_()

    y_torch.backward(grad_out_torch)
    gx_torch = x_torch.grad.detach().cpu().numpy()

    assert_allclose(gx_slow, gx_fast, name="grad_x slow vs fast")
    assert_allclose(gx_fast, gx_torch, name="grad_x fast vs torch")
    assert_allclose(gx_slow, gx_torch, name="grad_x slow vs torch")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(9999)
    torch.manual_seed(9999)

    print(f"[maxpool_cross_verify] Running {NUM_TRIALS} random trials...\n")

    try:
        for i in range(NUM_TRIALS):
            cross_verify_once(i)
            print(f"  [OK] trial {i+1}/{NUM_TRIALS}")
    except AssertionError as e:
        print(f"\n[FAILED] trial {i}: {e}")
        raise
    else:
        print("\n[maxpool_cross_verify] All max-pool tests passed!")
