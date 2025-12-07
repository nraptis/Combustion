# scorch/scorch_conv2d.py
from __future__ import annotations

import numpy as np
from scorch.scorch_module import ScorchModule


class ScorchConv2d(ScorchModule):
    """
    Pure loop-based reference implementation of a 2D convolution.

    Follows PyTorch semantics exactly:

        input:   (N, C_in,  H_in,  W_in)
        weight:  (C_out, C_in, K_h, K_w)
        bias:    (C_out,)
        output:  (N, C_out, H_out, W_out)

    H_out, W_out computed using the standard formula:

        H_out = floor((H_in + 2*pad_h - dilation_h*(K_h - 1) - 1) / stride_h + 1)
        W_out = floor((W_in + 2*pad_w - dilation_w*(K_w - 1) - 1) / stride_w + 1)

    All operations implemented via explicit nested python loops.
    Ideal for debugging and correctness verification.
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
            f"ScorchConv2d(in={in_channels}, out={out_channels}, "
            f"kernel=({self.k_h},{self.k_w}), stride=({self.s_h},{self.s_w}), "
            f"padding=({self.p_h},{self.p_w}), dilation=({self.d_h},{self.d_w}))"
        )

        # ----------------------------------------------------------
        # Parameter initialization (Kaiming-ish)
        # W shape: (C_out, C_in, K_h, K_w)
        # b shape: (C_out,)
        # ----------------------------------------------------------
        fan_in = self.in_channels * self.k_h * self.k_w
        limit = 1.0 / np.sqrt(fan_in)

        self.W = np.random.uniform(
            -limit, +limit,
            size=(self.out_channels, self.in_channels, self.k_h, self.k_w)
        ).astype(np.float32)

        self.b = np.zeros((self.out_channels,), dtype=np.float32)

        # Gradient buffers
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Cache for backward
        self._last_x = None
        self._last_output_shape = None

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray (N, C_in, H_in, W_in)
        returns: np.ndarray (N, C_out, H_out, W_out)
        """
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 4:
            raise ValueError(
                f"{self.name}: Expected 4D input (N,C,H,W), got {x.shape}"
            )

        N, C_in, H_in, W_in = x.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"{self.name}: Expected {self.in_channels} input channels, "
                f"got {C_in}"
            )

        # Compute output shape
        H_out = ((H_in + 2*self.p_h - self.d_h*(self.k_h - 1) - 1)
                 // self.s_h + 1)

        W_out = ((W_in + 2*self.p_w - self.d_w*(self.k_w - 1) - 1)
                 // self.s_w + 1)

        # Allocate output
        y = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)

        # Pad the input
        x_padded = np.pad(
            x,
            pad_width=(
                (0, 0),     # batch
                (0, 0),     # channels
                (self.p_h, self.p_h),
                (self.p_w, self.p_w),
            ),
            mode="constant",
            constant_values=0.0,
        )

        # ----------------------------------------------------------
        # Pure nested loops:
        #   for each batch, each output channel,
        #   each output pixel, each input channel,
        #   each kernel element...
        # ----------------------------------------------------------
        for n in range(N):
            for c_out in range(self.out_channels):
                for oy in range(H_out):
                    in_y_origin = oy * self.s_h
                    for ox in range(W_out):
                        in_x_origin = ox * self.s_w

                        acc = self.b[c_out]

                        for c_in in range(self.in_channels):
                            for ky in range(self.k_h):
                                iy = in_y_origin + ky * self.d_h
                                for kx in range(self.k_w):
                                    ix = in_x_origin + kx * self.d_w

                                    acc += (
                                        self.W[c_out, c_in, ky, kx] *
                                        x_padded[n, c_in, iy, ix]
                                    )

                        y[n, c_out, oy, ox] = acc

        # Save for backward
        self._last_x = x
        self._last_output_shape = (H_out, W_out)

        return y

    # --------------------------------------------------------------
    # Backward
    # --------------------------------------------------------------
    def backward(self, grad_out):
        """
        grad_out: np.ndarray (N, C_out, H_out, W_out)
        returns: grad_input (N, C_in, H_in, W_in)
        """
        if self._last_x is None:
            raise RuntimeError(f"{self.name}: backward called before forward.")

        x = self._last_x
        N, C_in, H_in, W_in = x.shape
        H_out, W_out = self._last_output_shape

        grad_out = np.asarray(grad_out, dtype=np.float32)

        # Allocate gradients
        grad_x = np.zeros_like(x, dtype=np.float32)
        self.grad_W.fill(0.0)
        self.grad_b.fill(0.0)

        # Pad x and grad_x so indexing aligns
        x_pad = np.pad(
            x,
            ((0,0),(0,0),(self.p_h,self.p_h),(self.p_w,self.p_w)),
            mode="constant",
        )
        grad_x_pad = np.zeros_like(x_pad, dtype=np.float32)

        # --------------------------------------------------------------
        # Compute gradients:
        #   grad_b[c_out] += grad_out[:,c_out,:,:]
        #   grad_W += x * grad_out
        #   grad_x += W * grad_out
        # --------------------------------------------------------------
        for n in range(N):
            for c_out in range(self.out_channels):
                for oy in range(H_out):
                    in_y_origin = oy * self.s_h
                    for ox in range(W_out):
                        in_x_origin = ox * self.s_w

                        go = grad_out[n, c_out, oy, ox]

                        # Bias gradient
                        self.grad_b[c_out] += go

                        # Weight + Input gradients
                        for c_in in range(self.in_channels):
                            for ky in range(self.k_h):
                                iy = in_y_origin + ky * self.d_h
                                for kx in range(self.k_w):
                                    ix = in_x_origin + kx * self.d_w

                                    # dL/dW
                                    self.grad_W[c_out, c_in, ky, kx] += (
                                        x_pad[n, c_in, iy, ix] * go
                                    )

                                    # dL/dx
                                    grad_x_pad[n, c_in, iy, ix] += (
                                        self.W[c_out, c_in, ky, kx] * go
                                    )

        # Remove padding from grad_x
        grad_x = grad_x_pad[
            :,
            :,
            self.p_h:self.p_h+H_in,
            self.p_w:self.p_w+W_in,
        ]

        return grad_x

    # --------------------------------------------------------------
    # Parameter handling
    # --------------------------------------------------------------
    def parameters(self):
        return [self.W, self.b]

    def zero_grad(self):
        self.grad_W.fill(0.0)
        self.grad_b.fill(0.0)

    def __repr__(self):
        return self.name
