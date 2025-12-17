import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dWithFrames(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
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

        # init similar-ish to torch default
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_ch * kh * kw
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # debug controls
        self.debug_enabled = False
        self.emit = None          # callable(frame_tensor, meta_dict)
        self.ref_index = 0        # which sample in batch to visualize

        # nudge granularity: "out", "out_in", "tap"
        self.nudge_mode = "out_in"

    def forward(self, x):
        # Fast, real output (keeps autograd + performance)
        y = F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding, dilation=self.dilation
        )

        # Optional debug cinema path
        if self.debug_enabled and self.emit is not None:
            with torch.no_grad():
                x_ref = x[self.ref_index:self.ref_index+1]  # shape (1, C, H, W)
                self._emit_manual_conv_frames(x_ref)

        return y

    def _emit_manual_conv_frames(self, x_ref):
        # Unfold turns sliding windows into columns.
        # patches: (1, C*kh*kw, L) where L = out_h*out_w
        kh, kw = self.weight.shape[2], self.weight.shape[3]
        patches = F.unfold(
            x_ref, kernel_size=(kh, kw),
            dilation=self.dilation, padding=self.padding, stride=self.stride
        )
        # weights reshaped: (out_ch, C*kh*kw)
        W = self.weight.view(self.weight.shape[0], -1)

        out_ch = W.shape[0]
        K = W.shape[1]              # C*kh*kw
        L = patches.shape[-1]       # number of output positions

        # accumulator in unfolded (out_ch, L)
        acc = torch.zeros(out_ch, L, device=x_ref.device, dtype=x_ref.dtype)

        # Helper to emit in (out_ch, out_h, out_w)
        def emit_frame(tag, meta_extra=None):
            # Recover output spatial dims:
            # L = out_h*out_w; infer using conv formula is possible, but easiest:
            # use F.conv2d once on ref to get shape (no grad here)
            y_shape = F.conv2d(
                x_ref, self.weight, self.bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation
            ).shape
            _, _, out_h, out_w = y_shape

            frame = acc.view(out_ch, out_h, out_w).detach().cpu()
            meta = {"tag": tag, "mode": self.nudge_mode}
            if meta_extra:
                meta.update(meta_extra)
            self.emit(frame, meta)

        # Start frame
        emit_frame("start")

        if self.nudge_mode == "out":
            # accumulate full kernel for each output channel, emit after each oc
            for oc in range(out_ch):
                acc[oc, :] = (W[oc:oc+1, :] @ patches[0, :, :]).squeeze(0)
                if self.bias is not None:
                    acc[oc, :] += self.bias[oc]
                emit_frame("after_out_channel", {"oc": oc})

        elif self.nudge_mode == "out_in":
            # emit after each (out_channel, kblock) contribution.
            # Here kblock can represent an input channel slice if you want,
            # but simplest is emit after each input-channel chunk:
            in_ch = self.weight.shape[1]
            k_per_in = kh * kw

            for oc in range(out_ch):
                for ic in range(in_ch):
                    k0 = ic * k_per_in
                    k1 = k0 + k_per_in
                    # partial dot product for this input channel's taps
                    acc[oc, :] += (W[oc, k0:k1].unsqueeze(0) @ patches[0, k0:k1, :]).squeeze(0)
                    emit_frame("after_out_in", {"oc": oc, "ic": ic})

                if self.bias is not None:
                    acc[oc, :] += self.bias[oc]
                    emit_frame("after_bias", {"oc": oc})

        elif self.nudge_mode == "tap":
            # emit after every single tap multiply-accumulate (very granular)
            for oc in range(out_ch):
                for k in range(K):
                    acc[oc, :] += W[oc, k] * patches[0, k, :]
                    emit_frame("after_tap", {"oc": oc, "k": k})
                if self.bias is not None:
                    acc[oc, :] += self.bias[oc]
                    emit_frame("after_bias", {"oc": oc})

        else:
            raise ValueError(f"Unknown nudge_mode: {self.nudge_mode}")
