from __future__ import annotations

# cifar_conv_2d_with_frames.py

from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarConv2dWithFrames(nn.Module):
    """
    A Conv2d-compatible module that:
      1) Computes the real output fast using F.conv2d (keeps autograd).
      2) Optionally runs a manual "nudge-by-nudge" accumulation on ONE reference image
         and calls `emit(frame_tensor, meta_dict)` for video frame generation.

    emit signature:
      emit(frame_tensor: torch.Tensor, meta: Dict)
        - frame_tensor is (out_ch, out_h, out_w) on CPU
        - meta includes info like tag/mode/oc/ic/k etc.

    emit_mode:
      - "snapshot": emit ONE full-frame (all out_ch) per forward
      - "nudge"   : emit many partial frames per forward (based on nudge_mode)

    nudge_mode:
      - "out"    : emit after each output channel is fully accumulated (few frames)
      - "out_in" : emit after each (out_channel, in_channel) chunk (moderate frames)
      - "tap"    : emit after every tap in the kernel (many frames)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ):
        super().__init__()

        # "snapshot" or "nudge"
        self.emit_mode: str = "snapshot"

        if isinstance(k, int):
            kh = kw = k
        else:
            kh, kw = k

        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        # Match Conv2d parameter shapes: (out_ch, in_ch, kh, kw)
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_ch)) if bias else None

        # Init similar-ish to torch default
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_ch * kh * kw
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # Debug controls (runtime only; not meant for checkpoints)
        self.debug_enabled: bool = False
        self.emit: Optional[Callable[[torch.Tensor, Dict], None]] = None
        self.ref_index: int = 0

        # Only used when emit_mode == "nudge"
        self.nudge_mode: str = "out_in"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast, real output (keeps autograd + performance)
        y = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # Optional debug cinema path (only when explicitly wired)
        if self.debug_enabled and (self.emit is not None):
            with torch.no_grad():
                x_ref = x[self.ref_index : self.ref_index + 1]  # (1,C,H,W)

                if self.emit_mode == "snapshot":
                    self._emit_snapshot(x_ref)
                elif self.emit_mode == "nudge":
                    self._emit_manual_conv_frames(x_ref)
                else:
                    raise ValueError(f"Unknown emit_mode: {self.emit_mode}")

        return y

    def _emit_snapshot(self, x_ref: torch.Tensor) -> None:
        """
        Emit ONE frame containing the FULL conv output (all out_ch)
        for the reference image.
        """
        if self.emit is None:
            return

        y_ref = F.conv2d(
            x_ref,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )  # (1, out_ch, out_h, out_w)

        # Ensure GPU/MPS work is finished before CPU read
        if x_ref.is_cuda:
            torch.cuda.synchronize()
        elif getattr(x_ref, "is_mps", False):
            torch.mps.synchronize()

        frame = y_ref[0].detach().cpu()  # (out_ch, out_h, out_w)

        meta = {
            "tag": "tick_snapshot",
            "mode": "snapshot",
        }

        self.emit(frame, meta)

    def _emit_manual_conv_frames(self, x_ref: torch.Tensor) -> None:
        if self.emit is None:
            return

        # Unfold turns sliding windows into columns.
        # patches: (1, C*kh*kw, L) where L = out_h*out_w
        kh, kw = self.weight.shape[2], self.weight.shape[3]
        patches = F.unfold(
            x_ref,
            kernel_size=(kh, kw),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

        # weights reshaped: (out_ch, C*kh*kw)
        W = self.weight.view(self.weight.shape[0], -1)

        out_ch = W.shape[0]
        K = W.shape[1]             # C*kh*kw
        L = patches.shape[-1]      # number of output positions

        # accumulator in unfolded space (out_ch, L)
        acc = torch.zeros(out_ch, L, device=x_ref.device, dtype=x_ref.dtype)

        # Determine output spatial dims once
        y_shape = F.conv2d(
            x_ref,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        ).shape
        _, _, out_h, out_w = y_shape

        def emit_frame(tag: str, meta_extra: Optional[Dict] = None) -> None:
            if self.emit is None:
                return
            frame = acc.view(out_ch, out_h, out_w).detach().cpu()
            meta: Dict = {"tag": tag, "mode": self.nudge_mode}
            if meta_extra:
                meta.update(meta_extra)
            self.emit(frame, meta)

        emit_frame("start")

        if self.nudge_mode == "out":
            for oc in range(out_ch):
                acc[oc, :] = (W[oc : oc + 1, :] @ patches[0, :, :]).squeeze(0)
                if self.bias is not None:
                    acc[oc, :] += self.bias[oc]
                emit_frame("after_out_channel", {"oc": oc})

        elif self.nudge_mode == "out_in":
            in_ch = self.weight.shape[1]
            k_per_in = kh * kw

            for oc in range(out_ch):
                for ic in range(in_ch):
                    k0 = ic * k_per_in
                    k1 = k0 + k_per_in
                    acc[oc, :] += (
                        W[oc, k0:k1].unsqueeze(0) @ patches[0, k0:k1, :]
                    ).squeeze(0)
                    emit_frame("after_out_in", {"oc": oc, "ic": ic})

                if self.bias is not None:
                    acc[oc, :] += self.bias[oc]
                    emit_frame("after_bias", {"oc": oc})

        elif self.nudge_mode == "tap":
            for oc in range(out_ch):
                for k in range(K):
                    acc[oc, :] += W[oc, k] * patches[0, k, :]
                    emit_frame("after_tap", {"oc": oc, "k": k})
                if self.bias is not None:
                    acc[oc, :] += self.bias[oc]
                    emit_frame("after_bias", {"oc": oc})

        else:
            raise ValueError(f"Unknown nudge_mode: {self.nudge_mode}")
