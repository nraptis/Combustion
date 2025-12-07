# scorch/convo_cross_verify.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scorch.scorch_conv2d import ScorchConv2d
from scorch.scorch_conv2d_fast import ScorchConv2dFast


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

NUM_TRIALS = 256   # convolution is heavier than linear tests


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
# Random input generators
# ------------------------------------------------------------

def random_int(low, high):
    return int(np.random.randint(low, high + 1))


def make_random_conv_params():
    """
    Random (but valid) convolution hyperparameters.

    Ensures:
      - kernel size <= padded input size
      - H_out >= 1 and W_out >= 1
        so that PyTorch Conv2d will accept the configuration.
    """
    while True:
        in_ch = random_int(1, 4)
        out_ch = random_int(1, 4)

        H = random_int(5, 15)
        W = random_int(5, 15)

        k_h = random_int(1, 9)
        k_w = random_int(1, 9)

        stride_h = random_int(1, 2)
        stride_w = random_int(1, 2)

        pad_h = random_int(0, 3)
        pad_w = random_int(0, 3)

        d_h = random_int(1, 2)
        d_w = random_int(1, 2)

        batch = random_int(1, 4)

        # --- Hard validity checks ---

        # 1) Kernel must fit in padded input
        if k_h > H + 2 * pad_h or k_w > W + 2 * pad_w:
            continue

        # 2) Output size must be >= 1x1 (PyTorch requirement)
        H_out = ((H + 2 * pad_h - d_h * (k_h - 1) - 1) // stride_h) + 1
        W_out = ((W + 2 * pad_w - d_w * (k_w - 1) - 1) // stride_w) + 1

        if H_out <= 0 or W_out <= 0:
            continue

        # If we get here, we have a valid configuration
        return {
            "N": batch,
            "C_in": in_ch,
            "H": H,
            "W": W,
            "C_out": out_ch,
            "kernel": (k_h, k_w),
            "stride": (stride_h, stride_w),
            "padding": (pad_h, pad_w),
            "dilation": (d_h, d_w),
        }


def make_random_input(N, C, H, W):
    return np.random.uniform(-3, 3, size=(N, C, H, W)).astype(np.float32)


# ------------------------------------------------------------
# Core test
# ------------------------------------------------------------

def cross_verify_once(trial_index: int):
    p = make_random_conv_params()

    N      = p["N"]
    C_in   = p["C_in"]
    H      = p["H"]
    W      = p["W"]
    C_out  = p["C_out"]
    ksize  = p["kernel"]
    stride = p["stride"]
    pad    = p["padding"]
    dial   = p["dilation"]

    # Create Scorch layers
    conv_slow = ScorchConv2d(
        C_in, C_out,
        kernel_size=ksize,
        stride=stride,
        padding=pad,
        dilation=dial,
    )

    conv_fast = ScorchConv2dFast(
        C_in, C_out,
        kernel_size=ksize,
        stride=stride,
        padding=pad,
        dilation=dial,
    )

    # Create PyTorch reference conv
    conv_torch = nn.Conv2d(
        C_in, C_out,
        kernel_size=ksize,
        stride=stride,
        padding=pad,
        dilation=dial,
        bias=True,
    )

    # --------------------------------------------------------
    # Sync weights & biases for exact comparison
    # --------------------------------------------------------
    W_slow, b_slow = conv_slow.parameters()
    W_fast, b_fast = conv_fast.parameters()

    with torch.no_grad():
        conv_torch.weight[:] = torch.from_numpy(W_slow)
        conv_torch.bias[:]   = torch.from_numpy(b_slow)

    # Copy slow params → fast
    np.copyto(W_fast, W_slow)
    np.copyto(b_fast, b_slow)

    # --------------------------------------------------------
    # Random input
    # --------------------------------------------------------
    x = make_random_input(N, C_in, H, W)
    x_torch = torch.from_numpy(x.copy()).requires_grad_(True)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    y_slow = conv_slow.forward(x.copy())
    y_fast = conv_fast.forward(x.copy())
    y_torch = conv_torch(x_torch)
    y_np = y_torch.detach().numpy()

    assert_allclose(y_slow, y_fast, name="slow vs fast forward")
    assert_allclose(y_slow, y_np,   name="slow vs torch forward")

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------
    grad_out = np.random.randn(*y_slow.shape).astype(np.float32)
    grad_out_torch = torch.from_numpy(grad_out.copy())

    gx_slow = conv_slow.backward(grad_out.copy())
    gx_fast = conv_fast.backward(grad_out.copy())

    # Torch backward
    conv_torch.zero_grad()
    if x_torch.grad is not None:
        x_torch.grad.zero_()

    y_torch.backward(grad_out_torch)
    gx_torch = x_torch.grad.detach().numpy()

    assert_allclose(gx_slow, gx_fast, name="grad_x slow vs fast")
    assert_allclose(gx_slow, gx_torch, name="grad_x slow vs torch")

    # --------------------------------------------------------
    # Weight gradients
    # --------------------------------------------------------
    gW_slow = conv_slow.grad_W
    gb_slow = conv_slow.grad_b

    gW_fast = conv_fast.grad_W
    gb_fast = conv_fast.grad_b

    gW_torch = conv_torch.weight.grad.detach().numpy()
    gb_torch = conv_torch.bias.grad.detach().numpy()

    assert_allclose(gW_slow, gW_fast, name="grad_W slow vs fast")
    assert_allclose(gb_slow, gb_fast, name="grad_b slow vs fast")

    assert_allclose(gW_slow, gW_torch, name="grad_W slow vs torch")
    assert_allclose(gb_slow, gb_torch, name="grad_b slow vs torch")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(1337)
    torch.manual_seed(1337)

    print(f"[convo_cross_verify] Running {NUM_TRIALS} random trials...\n")

    try:
        for i in range(NUM_TRIALS):
            cross_verify_once(i)
            print(f"  [OK] trial {i+1}/{NUM_TRIALS}")
    except AssertionError as e:
        print(f"\n[FAILED] trial {i}: {e}")
        raise
    else:
        print("\n[convo_cross_verify] All convolution tests passed!")
