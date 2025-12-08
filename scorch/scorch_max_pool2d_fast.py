# scorch/scorch_max_pool2d_fast.py
from __future__ import annotations

import numpy as np
from scorch.scorch_module import ScorchModule
from scorch.scorch_conv2d_fast import col2im


def im2col_maxpool(
    x,
    k_h, k_w,
    s_h, s_w,
    p_h, p_w,
    d_h, d_w,
):
    """
    im2col specialized for max-pooling:

    Input:
        x: (N, C, H, W)

    Output:
        cols: (N, C * k_h * k_w, H_out * W_out)

    Uses padding with -inf so that padded regions never win the max.
    """
    x = np.asarray(x, dtype=np.float32)
    N, C, H, W = x.shape

    H_out = ((H + 2 * p_h - d_h * (k_h - 1) - 1) // s_h) + 1
    W_out = ((W + 2 * p_w - d_w * (k_w - 1) - 1) // s_w) + 1

    # Pad with -inf so padding is never chosen as max
    x_padded = np.pad(
        x,
        pad_width=(
            (0, 0),
            (0, 0),
            (p_h, p_h),
            (p_w, p_w),
        ),
        mode="constant",
        constant_values=-np.inf,
    )

    cols = np.zeros((N, C * k_h * k_w, H_out * W_out), dtype=np.float32)

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

                        cols[:, patch_idx, out_col] = x_padded[:, c, iy, ix]
                        patch_idx += 1

            out_col += 1

    return cols  # (N, C*k_h*k_w, H_out*W_out)


class ScorchMaxPool2dFast(ScorchModule):
    """
    High-performance 2D max pooling using im2col + NumPy ops.

    Semantics:

        Input:   (N, C, H_in, W_in)  or (C, H_in, W_in) for single sample
        Output:  (N, C, H_out, W_out) or (C, H_out, W_out)

    Parameters:
        kernel_size:  int or (k_h, k_w)
        stride:       int or (s_h, s_w) (defaults to kernel_size if None)
        padding:      int or (p_h, p_w)
        dilation:     int or (d_h, d_w)

    Forward:
        - Uses im2col_maxpool (padding=-inf) to build patches.
        - Takes max over each patch.

    Backward:
        - Gradient flows only to the location that held the max in each patch.
        - Implemented via scattering into a grad_cols matrix, then col2im.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        name: str | None = None,
    ):
        super().__init__()

        # Normalize kernel size
        if isinstance(kernel_size, int):
            self.k_h = self.k_w = kernel_size
        else:
            self.k_h, self.k_w = kernel_size

        # Normalize stride (default to kernel_size if None)
        if stride is None:
            self.s_h = self.k_h
            self.s_w = self.k_w
        elif isinstance(stride, int):
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
            f"ScorchMaxPool2dFast(kernel=({self.k_h},{self.k_w}), "
            f"stride=({self.s_h},{self.s_w}), "
            f"padding=({self.p_h},{self.p_w}), "
            f"dilation=({self.d_h},{self.d_w}))"
        )

        # Cache for backward
        self._last_input_shape = None    # (N, C, H, W)
        self._H_out = None
        self._W_out = None
        self._argmax = None              # (N, C, H_out*W_out), indices in [0, K)
        self._input_was_3d = False       # True if original input was (C,H,W)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray, shape (N, C, H, W) or (C, H, W)
        returns: np.ndarray, shape (N, C, H_out, W_out) or (C, H_out, W_out)
        """
        x_arr = np.asarray(x, dtype=np.float32)

        # Accept 3D (C,H,W) by promoting to batch size 1
        if x_arr.ndim == 3:
            self._input_was_3d = True
            x_arr = x_arr[np.newaxis, ...]  # (1, C, H, W)
        elif x_arr.ndim == 4:
            self._input_was_3d = False
        else:
            raise ValueError(
                f"{self.name}: Expected 3D or 4D input, got shape {x_arr.shape}"
            )

        N, C, H, W = x_arr.shape

        # Compute output dims (same formula as conv)
        H_out = ((H + 2 * self.p_h - self.d_h * (self.k_h - 1) - 1)
                 // self.s_h + 1)
        W_out = ((W + 2 * self.p_w - self.d_w * (self.k_w - 1) - 1)
                 // self.s_w + 1)

        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"{self.name}: Non-positive output size "
                f"H_out={H_out}, W_out={W_out} for input (H={H},W={W})"
            )

        # im2col for pooling (pads with -inf)
        cols = im2col_maxpool(
            x_arr,
            self.k_h, self.k_w,
            self.s_h, self.s_w,
            self.p_h, self.p_w,
            self.d_h, self.d_w,
        )  # (N, C*k_h*k_w, H_out*W_out)

        K = self.k_h * self.k_w
        P = H_out * W_out

        # Reshape to: (N, C, K, P)
        cols_reshaped = cols.reshape(N, C, K, P)

        # Max over kernel dimension K
        max_vals = cols_reshaped.max(axis=2)                          # (N, C, P)
        argmax_idx = cols_reshaped.argmax(axis=2).astype(np.int32)    # (N, C, P)

        # Reshape back to (N, C, H_out, W_out)
        y = max_vals.reshape(N, C, H_out, W_out)

        # Cache for backward
        self._last_input_shape = (N, C, H, W)
        self._H_out = H_out
        self._W_out = W_out
        self._argmax = argmax_idx

        if self._input_was_3d:
            # Return (C, H_out, W_out)
            return y[0]
        else:
            return y

    # ----------------------------------------------------------
    # Backward
    # ----------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output:
            dL/dy, shape (N, C, H_out, W_out) or (C, H_out, W_out)

        Returns:
            dL/dx, shape (N, C, H, W) or (C, H, W)
        """
        if (
            self._last_input_shape is None
            or self._argmax is None
            or self._H_out is None
            or self._W_out is None
        ):
            raise RuntimeError(f"{self.name}: backward called before forward.")

        N, C, H, W = self._last_input_shape
        H_out = self._H_out
        W_out = self._W_out
        P = H_out * W_out
        K = self.k_h * self.k_w

        g_out = np.asarray(grad_output, dtype=np.float32)

        # Normalize grad_output to 4D
        if self._input_was_3d:
            if g_out.ndim != 3:
                raise ValueError(
                    f"{self.name}: grad_output for 3D input must be 3D "
                    f"(C,H_out,W_out), got {g_out.shape}"
                )
            if g_out.shape != (C, H_out, W_out):
                raise ValueError(
                    f"{self.name}: grad_output shape {g_out.shape} does not match "
                    f"(C,H_out,W_out)=({C},{H_out},{W_out})"
                )
            g_out = g_out[np.newaxis, ...]  # (1, C, H_out, W_out)
        else:
            if g_out.ndim != 4:
                raise ValueError(
                    f"{self.name}: grad_output for 4D input must be 4D "
                    f"(N,C,H_out,W_out), got {g_out.shape}"
                )
            if g_out.shape != (N, C, H_out, W_out):
                raise ValueError(
                    f"{self.name}: grad_output shape {g_out.shape} does not match "
                    f"(N,C,H_out,W_out)=({N},{C},{H_out},{W_out})"
                )

        # Flatten grad_output to (N, C, P)
        g_out_flat = g_out.reshape(N, C, P)   # (N, C, P)
        argmax = self._argmax                 # (N, C, P)

        # Build grad_cols of shape (N, C*K, P)
        grad_cols = np.zeros((N, C * K, P), dtype=np.float32)

        # For each (n,c,p), route g_out_flat[n,c,p] to
        # grad_cols[n, c*K + argmax[n,c,p], p]
        for n in range(N):
            for c in range(C):
                for p in range(P):
                    k_idx = argmax[n, c, p]
                    go = g_out_flat[n, c, p]
                    grad_cols[n, c * K + k_idx, p] += go

        # Convert column gradients back to image-space gradients
        grad_x = col2im(
            grad_cols,
            (N, C, H, W),
            self.k_h, self.k_w,
            self.s_h, self.s_w,
            self.p_h, self.p_w,
            self.d_h, self.d_w,
        )

        if self._input_was_3d:
            # Return (C, H, W)
            return grad_x[0]
        else:
            return grad_x

    # ----------------------------------------------------------
    # Parameters / grads
    # ----------------------------------------------------------
    def parameters(self):
        # No learnable parameters
        return []

    def zero_grad(self):
        # No parameter gradients to reset
        pass

    def __repr__(self):
        return self.name
