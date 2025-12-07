# relu_cross_verify.py

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from scorch.scorch_relu import ScorchReLU


NUM_TRIALS = 255


def assert_allclose(a, b, atol=1e-4, rtol=1e-4):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.abs(a - b)
        raise AssertionError(
            f"Arrays differ: max diff={diff.max()}, atol={atol}, rtol={rtol}"
        )


def make_random_nd_input():
    """
    Generates an input with random shape:
      rank = 1 to 4
      each dimension = 1 to 10
    values in (-5, 5)
    """
    rank = int(np.random.randint(1, 5))  # 1D…4D
    shape = tuple(int(np.random.randint(1, 11)) for _ in range(rank))
    x = np.random.uniform(-5.0, 5.0, size=shape).astype(np.float32)
    return x


def cross_verify_relu_once(trial_index: int):
    scorch_relu = ScorchReLU()
    torch_relu = nn.ReLU()

    # --------------
    # Random input
    # --------------
    x = make_random_nd_input()
    x_torch = torch.from_numpy(x.copy()).requires_grad_(True)

    # --------------
    # Forward
    # --------------
    y1 = scorch_relu.forward(x.copy())
    y2_t = torch_relu(x_torch)
    y2 = y2_t.detach().numpy()

    assert_allclose(y1, y2)

    # --------------
    # Backward
    # --------------
    grad_out = np.random.randn(*x.shape).astype(np.float32)

    gx1 = scorch_relu.backward(grad_out.copy())

    torch_relu.zero_grad()
    if x_torch.grad is not None:
        x_torch.grad.zero_()

    y2_t.backward(torch.from_numpy(grad_out.copy()))
    gx2 = x_torch.grad.detach().numpy()

    assert_allclose(gx1, gx2)


def test_relu_cross_verify():
    np.random.seed(1234)
    torch.manual_seed(1234)
    for i in range(NUM_TRIALS):
        cross_verify_relu_once(i)


if __name__ == "__main__":
    print(f"[relu_cross_verify] Running {NUM_TRIALS} random trials...")
    np.random.seed(1234)
    torch.manual_seed(1234)

    try:
        for i in range(NUM_TRIALS):
            cross_verify_relu_once(i)
            print(f"  [OK] trial {i+1}/{NUM_TRIALS}")
    except AssertionError as e:
        print(f"[relu_cross_verify] FAILED on trial {i}: {e}")
        raise
    else:
        print("[relu_cross_verify] All ReLU tests passed. "
              "ScorchReLU == torch.nn.ReLU for all random ND inputs.")
