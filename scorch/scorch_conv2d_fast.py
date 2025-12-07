# scorch/scorch_conv2d_fast.py
from __future__ import annotations

import numpy as np
from scorch.scorch_module import ScorchModule


def im2col(
    x, 
    k_h, k_w, 
    s_h, s_w, 
    p_h, p_w, 
    d_h, d_w
):
    """
    Convert (N, C, H, W) → (N, C*k_h*k_w, H_out*W_out) using im2col.
    """

    N, C, H, W = x.shape

    H_out = ((H + 2*p_h - d_h*(k_h - 1) - 1) // s_h) + 1
    W_out = ((W + 2*p_w - d_w*(k_w - 1) - 1) // s_w) + 1

    # Pad input
    x_padded = np.pad(
        x,
        pad_width=(
            (0, 0),
            (0, 0),
            (p_h, p_h),
            (p_w, p_w)
        ),
        mode="constant"
    )

    # Allocate output matrix
    cols = np.zeros((N, C * k_h * k_w, H_out * W_out), dtype=np.float32)

    out_col = 0
    for oy in range(H_out):
        in_y_origin = oy * s_h
        for ox in range(W_out):
            in_x_origin = ox * s_w

            # Gather all kernel positions for this output pixel
            patch_idx = 0
            for c in range(C):
                for ky in range(k_h):
                    iy = in_y_origin + ky * d_h
                    for kx in range(k_w):
                        ix = in_x_origin + kx * d_w

                        cols[:, patch_idx, out_col] = x_padded[:, c, iy, ix]
                        patch_idx += 1

            out_col += 1

    return cols  # (N, C*k_h*k_w, H_out*W_out)


def col2im(
    cols, 
    x_shape, 
    k_h, k_w, 
    s_h, s_w, 
    p_h, p_w, 
    d_h, d_w
):
    """
    Inverse of im2col:
    Convert (N, C*k_h*k_w, H_out*W_out) → (N, C, H, W)
    by adding contributions back into padded input space.
    """

    N, C, H, W = x_shape

    H_out = ((H + 2*p_h - d_h*(k_h - 1) - 1) // s_h) + 1
    W_out = ((W + 2*p_w - d_w*(k_w - 1) - 1) // s_w) + 1

    x_padded = np.zeros((N, C, H + 2*p_h, W + 2*p_w), dtype=np.float32)

    out_col = 0
    for oy in range(H_out):
        in_y_origin = oy * s_h
        for ox in range(W_out):
            in_x_origin = ox * s_w

            patch_idx = 0
            for c in range(C):
                for ky in range(k_h):
                    iy = in_y_origin + ky * d_h
                    for kx in range(k_w):
                        ix = in_x_origin + kx * d_w

                        x_padded[:, c, iy, ix] += cols[:, patch_idx, out_col]
                        patch_idx += 1

            out_col += 1

    # Remove padding
    return x_padded[:, :, p_h:p_h+H, p_w:p_w+W]


class ScorchConv2dFast(ScorchModule):
    """
    High-performance im2col + matmul Conv2d.
    Produces identical results to ScorchConv2d (nested loop version),
    but runs dramatically faster.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        name: str | None = None,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # Normalize kernel size
        if isinstance(kernel_size, int):
            self.k_h = self.k_w = kernel_size
        else:
            self.k_h, self.k_w = kernel_size

        # Normalize stride
        if isinstance(stride, int):
            self.s_h = self.s_w = stride
        else:
            self.s_h, self.s_w = stride

        # Normalize padding
        if isinstance(padding, int):
            self.p_h = self.p_w = padding
        else:
            self.p_h, self.p_w = padding

        # Normalize dilation
        if isinstance(dilation, int):
            self.d_h = self.d_w = dilation
        else:
            self.d_h, self.d_w = dilation

        self.name = name or (
            f"ScorchConv2dFast(in={in_channels}, out={out_channels}, "
            f"kernel=({self.k_h},{self.k_w}), stride=({self.s_h},{self.s_w}), "
            f"padding=({self.p_h},{self.p_w}), dilation=({self.d_h},{self.d_w}))"
        )

        # Parameter initialization
        fan_in = self.in_channels * self.k_h * self.k_w
        limit = 1.0 / np.sqrt(fan_in)

        self.W = np.random.uniform(
            -limit, +limit,
            size=(self.out_channels, self.in_channels, self.k_h, self.k_w)
        ).astype(np.float32)

        self.b = np.zeros((self.out_channels,), dtype=np.float32)

        # Grad buffers
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Cache
        self._last_cols = None
        self._last_x_shape = None
        self._H_out = None
        self._W_out = None

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 4:
            raise ValueError(
                f"{self.name}: Expected 4D input (N,C,H,W), got {x.shape}"
            )

        N, C, H, W = x.shape

        # Compute output dims
        H_out = ((H + 2*self.p_h - self.d_h*(self.k_h - 1) - 1) // self.s_h) + 1
        W_out = ((W + 2*self.p_w - self.d_w*(self.k_w - 1) - 1) // self.s_w) + 1

        # Store shapes for backward
        self._last_x_shape = (N, C, H, W)
        self._H_out = H_out
        self._W_out = W_out

        # Convert input into column matrix
        cols = im2col(
            x,
            self.k_h, self.k_w,
            self.s_h, self.s_w,
            self.p_h, self.p_w,
            self.d_h, self.d_w
        )  # (N, C*k_h*k_w, H_out*W_out)

        self._last_cols = cols

        # Reshape weights for matmul
        W_mat = self.W.reshape(self.out_channels, -1)  # (C_out, C_in*k_h*k_w)

        # Allocate output
        y = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)

        # For each batch:
        for n in range(N):
            # matmul: (C_out, K) @ (K, H_out*W_out)
            y_n = W_mat @ cols[n] + self.b.reshape(-1, 1)
            y[n] = y_n.reshape(self.out_channels, H_out, W_out)

        return y

    # ----------------------------------------------------------
    # Backward
    # ----------------------------------------------------------
    def backward(self, grad_out):
        """
        grad_out: (N, C_out, H_out, W_out)
        returns: grad_x of shape (N, C_in, H, W)
        """
        if self._last_cols is None:
            raise RuntimeError(f"{self.name}: backward called before forward.")

        cols = self._last_cols
        N, C, H, W = self._last_x_shape
        H_out = self._H_out
        W_out = self._W_out

        grad_out = np.asarray(grad_out, dtype=np.float32)

        # Flatten grad_out for matmul
        grad_out_2d = grad_out.reshape(N, self.out_channels, H_out * W_out)

        # Reset grads
        self.grad_W.fill(0.0)
        self.grad_b.fill(0.0)

        # For matmul shapes
        W_mat = self.W.reshape(self.out_channels, -1)      # (C_out, K)
        grad_W_mat = np.zeros_like(W_mat)                  # (C_out, K)

        # Output gradients accumulate into params
        for n in range(N):
            go = grad_out_2d[n]   # (C_out, H_out*W_out)

            # grad_b
            self.grad_b += go.sum(axis=1)

            # grad_W_mat += go @ cols[n].T
            grad_W_mat += go @ cols[n].transpose(1, 0)

        # Reshape grad_W back
        self.grad_W = grad_W_mat.reshape(
            self.out_channels, self.in_channels, self.k_h, self.k_w
        )

        # --------------------------------------------------
        # Compute grad_x via W^T * grad_out, then col2im
        # --------------------------------------------------
        grad_cols = np.zeros_like(cols)

        W_mat_T = W_mat.T  # (K, C_out)

        for n in range(N):
            go = grad_out_2d[n]       # (C_out, H_out*W_out)
            grad_cols[n] = W_mat_T @ go   # (K, H_out*W_out)

        # Convert column gradients back to image shape
        grad_x = col2im(
            grad_cols,
            (N, C, H, W),
            self.k_h, self.k_w,
            self.s_h, self.s_w,
            self.p_h, self.p_w,
            self.d_h, self.d_w
        )

        return grad_x

    # ----------------------------------------------------------
    # Parameter handling
    # ----------------------------------------------------------
    def parameters(self):
        return [self.W, self.b]

    def zero_grad(self):
        self.grad_W.fill(0.0)
        self.grad_b.fill(0.0)

    def __repr__(self):
        return self.name
