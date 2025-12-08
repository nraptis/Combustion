# scorch/scorch_max_pool2d.py
from __future__ import annotations

import numpy as np
from scorch.scorch_module import ScorchModule


class ScorchMaxPool2d(ScorchModule):
    """
    Pure loop-based reference implementation of 2D max pooling.

    Semantics (matched to PyTorch nn.MaxPool2d):

        Input:   (N, C, H_in, W_in)  or (C, H_in, W_in) for single sample
        Output:  (N, C, H_out, W_out) or (C, H_out, W_out)

    Parameters:
        kernel_size:  int or (k_h, k_w)
        stride:       int or (s_h, s_w) (defaults to kernel_size if None)
        padding:      int or (p_h, p_w)
        dilation:     int or (d_h, d_w)

    Forward:
        - Pads with -inf (so padded region never wins max).
        - For each (n, c, oy, ox), takes max over the pooled window.

    Backward:
        - Gradient flows only to the location that held the max.
        - We store the max coordinates during forward and scatter grad back.
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
            f"ScorchMaxPool2d(kernel=({self.k_h},{self.k_w}), "
            f"stride=({self.s_h},{self.s_w}), "
            f"padding=({self.p_h},{self.p_w}), "
            f"dilation=({self.d_h},{self.d_w}))"
        )

        # Caches for backward
        self._last_input_shape: tuple[int, int, int, int] | None = None
        self._H_out: int | None = None
        self._W_out: int | None = None
        self._input_was_3d: bool = False

        # Max locations in padded coordinates:
        #   _max_y, _max_x: shape (N, C, H_out, W_out), ints
        self._max_y: np.ndarray | None = None
        self._max_x: np.ndarray | None = None

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray, shape (N, C, H, W) or (C, H, W)
        returns: np.ndarray, shape (N, C, H_out, W_out) or (C, H_out, W_out)
        """
        x_arr = np.asarray(x, dtype=np.float32)

        # Allow 3D (C,H,W) by promoting to N=1
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

        # Pad with -inf so padding never wins the max
        x_padded = np.pad(
            x_arr,
            pad_width=(
                (0, 0),                 # batch
                (0, 0),                 # channels
                (self.p_h, self.p_h),   # height
                (self.p_w, self.p_w),   # width
            ),
            mode="constant",
            constant_values=-np.inf,
        )

        # Allocate output and max location caches
        y = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        max_y = np.zeros((N, C, H_out, W_out), dtype=np.int32)
        max_x = np.zeros((N, C, H_out, W_out), dtype=np.int32)

        # ------------------------------------------------------
        # Nested loops: N, C, H_out, W_out, then kernel window
        # ------------------------------------------------------
        for n in range(N):
            for c in range(C):
                for oy in range(H_out):
                    in_y_origin = oy * self.s_h
                    for ox in range(W_out):
                        in_x_origin = ox * self.s_w

                        best_val = -np.inf
                        best_iy = 0
                        best_ix = 0

                        for ky in range(self.k_h):
                            iy = in_y_origin + ky * self.d_h
                            for kx in range(self.k_w):
                                ix = in_x_origin + kx * self.d_w

                                v = x_padded[n, c, iy, ix]
                                # IMPORTANT: strict ">" to match np.argmax
                                if v > best_val:
                                    best_val = v
                                    best_iy = iy
                                    best_ix = ix

                        y[n, c, oy, ox] = best_val
                        max_y[n, c, oy, ox] = best_iy
                        max_x[n, c, oy, ox] = best_ix

        # Cache for backward
        self._last_input_shape = (N, C, H, W)
        self._H_out = H_out
        self._W_out = W_out
        self._max_y = max_y
        self._max_x = max_x

        if self._input_was_3d:
            return y[0]  # (C, H_out, W_out)
        else:
            return y     # (N, C, H_out, W_out)

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
            or self._H_out is None
            or self._W_out is None
            or self._max_y is None
            or self._max_x is None
        ):
            raise RuntimeError(f"{self.name}: backward called before forward.")

        N, C, H, W = self._last_input_shape
        H_out = self._H_out
        W_out = self._W_out

        g_out = np.asarray(grad_output, dtype=np.float32)

        # Normalize grad_output shape to 4D
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

        # Grad wrt *padded* input
        grad_x_padded = np.zeros(
            (N, C, H + 2 * self.p_h, W + 2 * self.p_w),
            dtype=np.float32,
        )

        max_y = self._max_y
        max_x = self._max_x

        # Scatter each grad_output element to its argmax location
        for n in range(N):
            for c in range(C):
                for oy in range(H_out):
                    for ox in range(W_out):
                        iy = max_y[n, c, oy, ox]
                        ix = max_x[n, c, oy, ox]
                        grad_x_padded[n, c, iy, ix] += g_out[n, c, oy, ox]

        # Remove padding
        grad_x = grad_x_padded[
            :,
            :,
            self.p_h:self.p_h + H,
            self.p_w:self.p_w + W,
        ]

        if self._input_was_3d:
            return grad_x[0]  # (C, H, W)
        else:
            return grad_x     # (N, C, H, W)

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
