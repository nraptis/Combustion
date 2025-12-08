# scorch/scorch_adaptive_avg_pool2d.py
from __future__ import annotations

import numpy as np
from scorch.scorch_module import ScorchModule


class ScorchAdaptiveAvgPool2d(ScorchModule):
    """
    Adaptive average pooling for 2D inputs, matching PyTorch's behavior.

    Supports:
        - input:  (N, C, H, W)  → output: (N, C, H_out, W_out)
        - input:  (C, H, W)     → output: (C, H_out, W_out)

    Where (H_out, W_out) = output_size.

    For each output index (oh, ow), the pooling region in the input is:

        h_start = floor(oh     * H / H_out)
        h_end   = ceil ((oh+1) * H / H_out)

        w_start = floor(ow     * W / W_out)
        w_end   = ceil ((ow+1) * W / W_out)

    (implemented with exact integer math)

    Forward:
        y[n,c,oh,ow] = mean over x[n,c,h_start:h_end, w_start:w_end]

    Backward:
        dL/dx[n,c,h,w] = sum over all (oh,ow) whose region includes (h,w)
                         grad_out[n,c,oh,ow] / area_region(oh,ow)
    """

    def __init__(self, output_size, name: str | None = None):
        super().__init__()

        # Normalize output_size to (H_out, W_out)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        if not (isinstance(output_size, (tuple, list)) and len(output_size) == 2):
            raise ValueError(
                f"ScorchAdaptiveAvgPool2d: output_size must be int or (h,w) tuple, "
                f"got {output_size!r}"
            )

        H_out, W_out = int(output_size[0]), int(output_size[1])
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"ScorchAdaptiveAvgPool2d: output_size elements must be > 0, "
                f"got {output_size}"
            )

        self.output_size = (H_out, W_out)
        self.name = name or f"ScorchAdaptiveAvgPool2d(output_size={self.output_size})"

        # Caches
        self._last_input_shape: tuple[int, int, int, int] | None = None
        self._input_was_3d: bool = False

        # Per-forward index maps for backward reuse
        self._h_start = None
        self._h_end = None
        self._w_start = None
        self._w_end = None

    # ------------------------------------------------------
    # Internal helpers to build pooling regions
    # ------------------------------------------------------
    @staticmethod
    def _compute_region_indices(in_size: int, out_size: int):
        """
        Compute start/end indices for each output index, matching the
        "adaptive" logic used in frameworks like PyTorch.

        For each o in [0, out_size):
            start = floor(o     * in_size / out_size)
            end   = ceil ((o+1) * in_size / out_size)

        Implemented with integer arithmetic to avoid floating rounding issues:

            start = (o * in_size) // out_size
            end   = ((o + 1) * in_size + out_size - 1) // out_size
        """
        starts = []
        ends = []
        for o in range(out_size):
            start = (o * in_size) // out_size
            end = ((o + 1) * in_size + out_size - 1) // out_size
            # Clamp just in case
            start = max(0, min(start, in_size))
            end = max(start, min(end, in_size))
            starts.append(start)
            ends.append(end)
        return starts, ends

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray, shape (N,C,H,W) or (C,H,W)

        returns:
            y: shape (N,C,H_out,W_out) or (C,H_out,W_out)
        """
        x_arr = np.asarray(x, dtype=np.float32)

        if x_arr.ndim == 3:
            # (C,H,W) => (1,C,H,W)
            self._input_was_3d = True
            x_arr = x_arr[np.newaxis, ...]
        elif x_arr.ndim == 4:
            self._input_was_3d = False
        else:
            raise ValueError(
                f"{self.name}: Expected 3D or 4D input, got shape {x_arr.shape}"
            )

        N, C, H, W = x_arr.shape
        self._last_input_shape = (N, C, H, W)

        H_out, W_out = self.output_size

        # Precompute pooling regions
        h_start, h_end = self._compute_region_indices(H, H_out)
        w_start, w_end = self._compute_region_indices(W, W_out)

        self._h_start = h_start
        self._h_end = h_end
        self._w_start = w_start
        self._w_end = w_end

        # Allocate output
        y = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    hs = h_start[oh]
                    he = h_end[oh]
                    h_len = he - hs
                    if h_len <= 0:
                        continue

                    for ow in range(W_out):
                        ws = w_start[ow]
                        we = w_end[ow]
                        w_len = we - ws
                        if w_len <= 0:
                            continue

                        region = x_arr[n, c, hs:he, ws:we]
                        area = float(h_len * w_len)
                        y[n, c, oh, ow] = region.sum() / area

        if self._input_was_3d:
            return y[0]  # (C,H_out,W_out)
        else:
            return y     # (N,C,H_out,W_out)

    # ------------------------------------------------------
    # Backward
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output:
            - If input was 4D: (N,C,H_out,W_out)
            - If input was 3D: (C,H_out,W_out)

        returns:
            grad_input with same shape as original x:
            - (N,C,H,W) or (C,H,W)
        """
        if self._last_input_shape is None:
            raise RuntimeError(f"{self.name}: backward called before forward.")

        if self._h_start is None or self._w_start is None:
            raise RuntimeError(f"{self.name}: region indices missing; forward not run?")

        N, C, H, W = self._last_input_shape
        H_out, W_out = self.output_size
        h_start = self._h_start
        h_end = self._h_end
        w_start = self._w_start
        w_end = self._w_end

        g_out = np.asarray(grad_output, dtype=np.float32)

        # Normalize grad_output shape to 4D
        if self._input_was_3d:
            if g_out.ndim != 3 or g_out.shape[0] != C:
                raise ValueError(
                    f"{self.name}: grad_output for 3D input must be (C,H_out,W_out), "
                    f"got {g_out.shape}"
                )
            g_out = g_out[np.newaxis, ...]  # (1,C,H_out,W_out)
        else:
            if g_out.ndim != 4 or g_out.shape != (N, C, H_out, W_out):
                raise ValueError(
                    f"{self.name}: grad_output for 4D input must be (N,C,H_out,W_out), "
                    f"got {g_out.shape}"
                )

        grad_x = np.zeros((N, C, H, W), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    hs = h_start[oh]
                    he = h_end[oh]
                    h_len = he - hs
                    if h_len <= 0:
                        continue

                    for ow in range(W_out):
                        ws = w_start[ow]
                        we = w_end[ow]
                        w_len = we - ws
                        if w_len <= 0:
                            continue

                        go = g_out[n, c, oh, ow]
                        area = float(h_len * w_len)
                        if area <= 0:
                            continue

                        share = go / area
                        for h_idx in range(hs, he):
                            for w_idx in range(ws, we):
                                grad_x[n, c, h_idx, w_idx] += share

        if self._input_was_3d:
            return grad_x[0]  # (C,H,W)
        else:
            return grad_x     # (N,C,H,W)

    # ------------------------------------------------------
    # Parameters / grads
    # ------------------------------------------------------
    def parameters(self):
        # No learnable parameters
        return []

    def zero_grad(self):
        # Nothing to reset
        pass

    def __repr__(self):
        return self.name
