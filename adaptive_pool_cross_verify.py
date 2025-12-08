# adaptive_pool_cross_verify.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scorch.scorch_adaptive_avg_pool2d import ScorchAdaptiveAvgPool2d


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

NUM_TRIALS_4D = 2048
NUM_TRIALS_3D = 2048


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


def random_int(low, high):
    return int(np.random.randint(low, high + 1))


# ------------------------------------------------------------
# Random param + input generators
# ------------------------------------------------------------

def make_random_pool_params():
    """
    Random (but always valid) parameters for AdaptiveAvgPool2d(1x1).

    We only vary the input shape:
      - batch size N      in [1, 4]
      - channels C        in [1, 16]
      - height / width    in [1, 64]
    """
    N = random_int(1, 4)
    C = random_int(1, 16)
    H = random_int(1, 64)
    W = random_int(1, 64)
    return {"N": N, "C": C, "H": H, "W": W}


def make_random_input_4d(N, C, H, W):
    # mid-scale random values
    return np.random.uniform(-5.0, 5.0, size=(N, C, H, W)).astype(np.float32)


def make_random_input_3d(C, H, W):
    return np.random.uniform(-5.0, 5.0, size=(C, H, W)).astype(np.float32)


# ------------------------------------------------------------
# Core tests
# ------------------------------------------------------------

def cross_verify_4d_once(trial_index: int):
    """
    Compare ScorchAdaptiveAvgPool2d vs nn.AdaptiveAvgPool2d
    on a 4D input: (N, C, H, W).
    """
    p = make_random_pool_params()
    N, C, H, W = p["N"], p["C"], p["H"], p["W"]

    D1 = random_int(1, 4)
    D2 = random_int(1, 4)

    # Create models
    pool_scorch = ScorchAdaptiveAvgPool2d(output_size=(D1, D2))
    pool_torch = nn.AdaptiveAvgPool2d((D1, D2))

    # Random input
    x_np = make_random_input_4d(N, C, H, W)
    x_torch = torch.from_numpy(x_np.copy()).requires_grad_(True)

    # Forward
    y_s = pool_scorch.forward(x_np)     # (N,C,1,1)
    y_t = pool_torch(x_torch)           # (N,C,1,1)
    y_t_np = y_t.detach().numpy()

    assert_allclose(
        y_s, y_t_np,
        name=f"forward 4D (trial {trial_index})"
    )

    # Backward: random grad on output
    grad_out_np = np.random.randn(*y_s.shape).astype(np.float32)
    grad_out_torch = torch.from_numpy(grad_out_np.copy())

    # Scorch backward
    grad_x_s = pool_scorch.backward(grad_out_np)  # (N,C,H,W)

    # Torch backward
    if x_torch.grad is not None:
        x_torch.grad.zero_()
    y_t.backward(grad_out_torch)
    grad_x_t = x_torch.grad.detach().numpy()      # (N,C,H,W)

    assert_allclose(
        grad_x_s, grad_x_t,
        name=f"backward 4D grad_x (trial {trial_index})"
    )


def cross_verify_3d_once(trial_index: int):
    """
    Compare ScorchAdaptiveAvgPool2d vs nn.AdaptiveAvgPool2d
    on a 3D input for Scorch: (C, H, W).

    PyTorch AdaptiveAvgPool2d only accepts 4D (N,C,H,W),
    so we wrap the input with N=1 on the PyTorch side.
    """
    # Reuse the same ranges but ignore N, use N=1 instead
    C = random_int(1, 16)
    H = random_int(1, 64)
    W = random_int(1, 64)

    pool_scorch = ScorchAdaptiveAvgPool2d(output_size=(1, 1))
    pool_torch = nn.AdaptiveAvgPool2d((1, 1))

    # Random input
    x_np_3d = make_random_input_3d(C, H, W)        # (C,H,W)
    x_np_4d = x_np_3d[np.newaxis, ...]             # (1,C,H,W) for torch

    x_torch = torch.from_numpy(x_np_4d.copy()).requires_grad_(True)

    # Forward
    y_s = pool_scorch.forward(x_np_3d)             # (C,1,1)
    y_t = pool_torch(x_torch)                      # (1,C,1,1)
    y_t_np_3d = y_t.detach().numpy()[0]            # (C,1,1)

    assert_allclose(
        y_s, y_t_np_3d,
        name=f"forward 3D (trial {trial_index})"
    )

    # Backward: random grad in 3D form for Scorch
    grad_out_3d = np.random.randn(*y_s.shape).astype(np.float32)   # (C,1,1)
    grad_out_4d = grad_out_3d[np.newaxis, ...]                     # (1,C,1,1)

    # Scorch backward
    grad_x_s_3d = pool_scorch.backward(grad_out_3d)                # (C,H,W)

    # Torch backward
    grad_out_torch = torch.from_numpy(grad_out_4d.copy())
    if x_torch.grad is not None:
        x_torch.grad.zero_()
    y_t.backward(grad_out_torch)
    grad_x_t_4d = x_torch.grad.detach().numpy()                    # (1,C,H,W)
    grad_x_t_3d = grad_x_t_4d[0]                                   # (C,H,W)

    assert_allclose(
        grad_x_s_3d, grad_x_t_3d,
        name=f"backward 3D grad_x (trial {trial_index})"
    )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(999)
    torch.manual_seed(1337)

    print(f"[adaptive_pool_cross_verify] Running {NUM_TRIALS_4D} trials for 4D inputs...")
    try:
        for i in range(NUM_TRIALS_4D):
            cross_verify_4d_once(i)
            print(f"  [OK] 4D trial {i+1}/{NUM_TRIALS_4D}")
    except AssertionError as e:
        print(f"\n[FAILED] 4D trial {i}: {e}")
        raise

    print(f"\n[adaptive_pool_cross_verify] Running {NUM_TRIALS_3D} trials for 3D inputs...")
    try:
        for i in range(NUM_TRIALS_3D):
            cross_verify_3d_once(i)
            print(f"  [OK] 3D trial {i+1}/{NUM_TRIALS_3D}")
    except AssertionError as e:
        print(f"\n[FAILED] 3D trial {i}: {e}")
        raise

    print("\n[adaptive_pool_cross_verify] All AdaptiveAvgPool2d tests passed!")
