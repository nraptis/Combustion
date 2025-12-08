# scorch/scorch_small_convnet_multitask.py
from __future__ import annotations

import numpy as np

from scorch.scorch_module import ScorchModule
from scorch.scorch_sequential import ScorchSequential
from scorch.scorch_conv2d_fast import ScorchConv2dFast
from scorch.scorch_max_pool2d_fast import ScorchMaxPool2dFast
from scorch.scorch_relu import ScorchReLU
from scorch.scorch_linear import ScorchLinear
from scorch.scorch_adaptive_avg_pool2d import ScorchAdaptiveAvgPool2d


class ScorchSmallConvNetMultiTask(ScorchModule):
    """
    Tiny CNN that:
      - Produces class logits (for patch classification).
      - Produces a per-pixel mask (for segmentation).

    Architecture mirrors the PyTorch reference:

        features:
            Conv2d(in_channels, 8,  kernel_size=3, padding=1)
            ReLU
            MaxPool2d(2)
            Conv2d(8, 16, kernel_size=3, padding=1)
            ReLU
            MaxPool2d(2)

        classifier:
            AdaptiveAvgPool2d((1,1))
            Flatten (N,16,1,1) -> (N,16)
            Linear(16, 64)
            ReLU
            Linear(64, num_classes)

        seg_head:
            Conv2d(16, 16, kernel_size=3, padding=1)
            ReLU
            Conv2d(16, 1,  kernel_size=1)

    Forward(x):
        - x: np.ndarray, shape (N, in_channels, H, W)
        Returns:
        - class_logits: (N, num_classes)
        - seg_logits:   (N, 1, Hf, Wf) where Hf,Wf are features' spatial dims

    Backward expects a tuple:
        backward((grad_class_logits, grad_seg_logits)).

    Returns:
        grad_input: dL/dx, shape (N, in_channels, H, W)
    """

    def __init__(self, in_channels: int, num_classes: int, name: str | None = None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.name = name or f"ScorchSmallConvNetMultiTask(in={in_channels}, classes={num_classes})"

        # --------------------------------------------------
        # Shared feature extractor
        # --------------------------------------------------
        self.features = ScorchSequential(
            ScorchConv2dFast(
                in_channels=self.in_channels,
                out_channels=8,
                kernel_size=3,
                padding=1,
                name="conv1",
            ),
            ScorchReLU(name="relu1"),
            ScorchMaxPool2dFast(kernel_size=2, stride=2, name="pool1"),

            ScorchConv2dFast(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
                name="conv2",
            ),
            ScorchReLU(name="relu2"),
            ScorchMaxPool2dFast(kernel_size=2, stride=2, name="pool2"),
        )

        # --------------------------------------------------
        # Classification head
        #   feats -> AdaptiveAvgPool2d(1,1) -> Flatten -> FC(16->64) -> ReLU -> FC(64->num_classes)
        # --------------------------------------------------
        self.pool = ScorchAdaptiveAvgPool2d(output_size=(1, 1), name="global_avg_pool")

        # 16 channels after conv2
        self.classifier = ScorchSequential(
            ScorchLinear(16, 64, name="cls_fc1"),
            ScorchReLU(name="cls_relu"),
            ScorchLinear(64, self.num_classes, name="cls_fc2"),
        )

        # We'll need to remember the pooled shape for reshaping in backward
        self._last_pooled_shape = None  # (N, 16, 1, 1)

        # --------------------------------------------------
        # Segmentation head
        #   feats -> Conv(16->16, k3,p1) -> ReLU -> Conv(16->1, k1)
        # --------------------------------------------------
        self.seg_head = ScorchSequential(
            ScorchConv2dFast(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1,
                name="seg_conv1",
            ),
            ScorchReLU(name="seg_relu"),
            ScorchConv2dFast(
                in_channels=16,
                out_channels=1,
                kernel_size=1,
                padding=0,
                name="seg_conv2",
            ),
        )

        # Cache last feature map for backward sanity/debug if needed
        self._last_feats_shape = None  # (N,16,Hf,Wf)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x):
        """
        x: np.ndarray, shape (N, in_channels, H, W)

        returns:
            class_logits: (N, num_classes)
            seg_logits:   (N, 1, Hf, Wf)
        """
        x_arr = np.asarray(x, dtype=np.float32)

        if x_arr.ndim != 4:
            raise ValueError(
                f"{self.name}: Expected 4D input (N,C,H,W), got {x_arr.shape}"
            )

        # Shared features
        feats = self.features.forward(x_arr)  # (N,16,Hf,Wf)
        self._last_feats_shape = feats.shape

        N, C_feat, Hf, Wf = feats.shape
        if C_feat != 16:
            raise ValueError(
                f"{self.name}: Expected features to have 16 channels, got {C_feat}"
            )

        # --------------------------------------------------
        # Classification branch
        # --------------------------------------------------
        pooled = self.pool.forward(feats)  # (N,16,1,1)
        self._last_pooled_shape = pooled.shape

        # Flatten: (N,16,1,1) -> (N,16)
        pooled_flat = pooled.reshape(N, C_feat)

        class_logits = self.classifier.forward(pooled_flat)  # (N,num_classes)

        # --------------------------------------------------
        # Segmentation branch
        # --------------------------------------------------
        seg_logits = self.seg_head.forward(feats)  # (N,1,Hf,Wf)

        return class_logits, seg_logits

    # ------------------------------------------------------
    # Backward
    # ------------------------------------------------------
    def backward(self, grad_output):
        """
        grad_output: tuple (grad_class_logits, grad_seg_logits):

            grad_class_logits: shape (N, num_classes)
            grad_seg_logits:   shape (N, 1, Hf, Wf)

        returns:
            grad_input: shape (N, in_channels, H, W)
        """
        if not isinstance(grad_output, (tuple, list)) or len(grad_output) != 2:
            raise ValueError(
                f"{self.name}.backward expects (grad_class_logits, grad_seg_logits) tuple, "
                f"got {type(grad_output)}"
            )

        grad_class_logits, grad_seg_logits = grad_output

        if self._last_feats_shape is None or self._last_pooled_shape is None:
            raise RuntimeError(
                f"{self.name}: backward called before forward (missing caches)."
            )

        N, C_feat, Hf, Wf = self._last_feats_shape
        pooled_shape = self._last_pooled_shape  # (N,16,1,1)

        grad_class = np.asarray(grad_class_logits, dtype=np.float32)
        grad_seg = np.asarray(grad_seg_logits, dtype=np.float32)

        # --------------------------------------------------
        # Classification branch backward
        # --------------------------------------------------
        # classifier: Linear(16->64) -> ReLU -> Linear(64->num_classes)
        # forward got pooled_flat (N,16)
        grad_pooled_flat = self.classifier.backward(grad_class)  # (N,16)

        # Unflatten back to pooled shape (N,16,1,1)
        grad_pooled = grad_pooled_flat.reshape(pooled_shape)

        # Back through AdaptiveAvgPool2d
        grad_feats_from_cls = self.pool.backward(grad_pooled)  # (N,16,Hf,Wf)

        # --------------------------------------------------
        # Segmentation branch backward
        # --------------------------------------------------
        grad_feats_from_seg = self.seg_head.backward(grad_seg)  # (N,16,Hf,Wf)

        # --------------------------------------------------
        # Combine gradients from both branches
        # --------------------------------------------------
        grad_feats_total = grad_feats_from_cls + grad_feats_from_seg  # (N,16,Hf,Wf)

        # Back through shared features
        grad_input = self.features.backward(grad_feats_total)  # (N,in_channels,H,W)

        return grad_input

    # ------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------
    def parameters(self):
        params = []
        params.extend(self.features.parameters())
        params.extend(self.classifier.parameters())
        params.extend(self.seg_head.parameters())
        return params

    def zero_grad(self):
        self.features.zero_grad()
        self.classifier.zero_grad()
        self.seg_head.zero_grad()

    def __repr__(self):
        return self.name
