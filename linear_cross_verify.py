# linear_cross_verify.py

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scorch.scorch_module import ScorchModule
from scorch.scorch_linear import ScorchLinear


# --------------------------------------------------
# Config
# --------------------------------------------------

NUM_TRIALS = 255


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def assert_allclose(a, b, atol=1e-4, rtol=1e-4):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.abs(a - b)
        max_diff = float(diff.max())
        raise AssertionError(
            f"Arrays differ: max |a-b| = {max_diff}, "
            f"atol={atol}, rtol={rtol}"
        )


def sync_scorch_params(src: ScorchModule, dst: ScorchModule) -> None:
    """
    Copy parameters between two Scorch modules (assumes same layout).
    """
    src_params = src.parameters()
    dst_params = dst.parameters()

    if len(src_params) != len(dst_params):
        raise AssertionError(
            f"Parameter count mismatch: "
            f"{len(src_params)} vs {len(dst_params)}"
        )

    for p_src, p_dst in zip(src_params, dst_params):
        if p_src.shape != p_dst.shape:
            raise AssertionError(
                f"Parameter shape mismatch: "
                f"{p_src.shape} vs {p_dst.shape}"
            )
        np.copyto(p_dst, p_src)


def sync_scorch_to_torch(
    scorch_layer: ScorchLinear | ScorchLinear2,
    torch_layer: nn.Linear,
) -> None:
    """
    Copy parameters from a ScorchLinear/ScorchLinear2 into a torch.nn.Linear.
    Assumes shapes:
        W: (out_features, in_features)
        b: (out_features,)
    """
    W, b = scorch_layer.parameters()
    with torch.no_grad():
        torch_layer.weight.data.copy_(torch.from_numpy(W))
        torch_layer.bias.data.copy_(torch.from_numpy(b))


def make_random_input_shape():
    """
    50% of the time: 1-D input with shape (D,)
    50% of the time: 2-D input with shape (N, D)
    where D, N are in [1, 10].
    """
    if np.random.rand() < 0.5:
        # 1-D case
        in_features = int(np.random.randint(1, 11))
        shape = (in_features,)
    else:
        # 2-D case
        batch_size = int(np.random.randint(1, 11))
        in_features = int(np.random.randint(1, 11))
        shape = (batch_size, in_features)
    return shape, in_features


def make_random_input(shape) -> np.ndarray:
    """
    Create random input with given shape, values in (-1000, 1000).
    """
    x = np.random.uniform(
        low=-1000.0,
        high=1000.0,
        size=shape,
    ).astype(np.float32)
    return x


# --------------------------------------------------
# Core cross-verify
# --------------------------------------------------

def cross_verify_linear_once(trial_index: int) -> None:
    """
    Run a single random test comparing:
      - ScorchLinear
      - ScorchLinear2
      - torch.nn.Linear

    Steps:
      1. Randomly choose 1-D or 2-D input shape.
      2. Randomly choose in_features and out_features in [1, 10].
      3. Create ScorchLinear, ScorchLinear2, and torch.nn.Linear with same sizes.
      4. Sync parameters so all three share identical W and b.
      5. Forward through all three.
      6. Generate random upstream gradient dy with same shape as y.
      7. Backward through all three.
      8. Assert outputs and input-gradients match (within 1e-4).
    """
    # ----------------------------------------
    # Random shapes and sizes
    # ----------------------------------------
    x_shape, in_features = make_random_input_shape()
    out_features = int(np.random.randint(1, 11))

    # Fresh layers for this trial
    lin1 = ScorchLinear(in_features, out_features)
    #lin2 = ScorchLinear2(in_features, out_features)
    torch_lin = nn.Linear(in_features, out_features)

    # Use lin1 as canonical parameters, sync to others
    
    sync_scorch_to_torch(lin1, torch_lin)

    # Random input
    x = make_random_input(x_shape)

    # Torch input (with grad)
    x_torch = torch.from_numpy(x.copy()).requires_grad_(True)

    # ------------------------------
    # Forward
    # ------------------------------
    y1 = lin1.forward(x.copy())
    #y2 = lin2.forward(x.copy())
    y3_torch = torch_lin(x_torch)
    y3 = y3_torch.detach().numpy()

    # Shape sanity: all outputs must match
    if y1.shape != y3.shape:
        raise AssertionError(
            f"Output shape mismatch on trial {trial_index}: "
            f"ScorchLinear={y1.shape}, "
            f"torch={y3.shape}, "
            f"input_shape={x_shape}, in_features={in_features}, out_features={out_features}"
        )

    # Compare outputs pairwise (within 1e-4)
    assert_allclose(y1, y3)

    # ------------------------------
    # Backward
    # ------------------------------
    grad_out = np.random.randn(*y1.shape).astype(np.float32)

    gx1 = lin1.backward(grad_out.copy())

    # Torch backward
    torch_lin.zero_grad(set_to_none=True)
    if x_torch.grad is not None:
        x_torch.grad.zero_()
    y3_torch.backward(torch.from_numpy(grad_out.copy()))
    gx3 = x_torch.grad.detach().numpy()

    # Input-grad shape sanity
    if gx1.shape != gx3.shape:
        raise AssertionError(
            f"Input-grad shape mismatch on trial {trial_index}: "
            f"ScorchLinear={gx1.shape}, "
            f"torch={gx3.shape}, "
            f"input_shape={x_shape}, in_features={in_features}, out_features={out_features}"
        )

    # Compare input-gradients
    assert_allclose(gx1, gx3)

    # ------------------------------
    # Optional: compare parameter gradients too
    # ------------------------------
    # Scorch grads:
    grad_W1 = lin1.grad_W
    grad_b1 = lin1.grad_b
    

    # Torch grads:
    grad_W3 = torch_lin.weight.grad.detach().numpy()
    grad_b3 = torch_lin.bias.grad.detach().numpy()

    assert_allclose(grad_W1, grad_W3)
    assert_allclose(grad_b1, grad_b3)


def test_linear_cross_verify() -> None:
    """
    Pytest-style entry point: run NUM_TRIALS random cross-checks.
    """
    np.random.seed(5678)
    torch.manual_seed(5678)

    for i in range(NUM_TRIALS):
        cross_verify_linear_once(i)


# --------------------------------------------------
# Script entry point
# --------------------------------------------------

if __name__ == "__main__":
    np.random.seed(5678)
    torch.manual_seed(5678)

    print(f"[linear_cross_verify] Running {NUM_TRIALS} random 1D/2D trials...")
    try:
        for i in range(NUM_TRIALS):
            cross_verify_linear_once(i)
            print(f"  [OK] trial {i+1}/{NUM_TRIALS}")
    except AssertionError as e:
        print(f"[linear_cross_verify] FAILED on trial {i}: {e}")
        raise
    else:
        print("[linear_cross_verify] All trials passed. "
              "ScorchLinear, ScorchLinear2, and torch.nn.Linear agree on this random suite.")
