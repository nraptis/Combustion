# multitask_cross_verify.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from scorch.scorch_small_convnet_multitask import ScorchSmallConvNetMultiTask


# ------------------------------------------------------------
# PyTorch reference model
# ------------------------------------------------------------

class SmallConvNetMultiTask(nn.Module):
    """
    Tiny CNN that:
      - Produces class logits (for patch classification).
      - Produces a per-pixel mask (for segmentation).
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature extractor (same spirit as SmallConvNet)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 1/2

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 1/4
        )

        # Classification head: global avg pool -> FC
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),         # (N,16,1,1) -> (N,16)
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        # Segmentation head: upsample features back to input size
        self.seg_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        feats = self.features(x)                  # (N,16,Hf,Wf)

        pooled = self.pool(feats)                 # (N,16,1,1)
        class_logits = self.classifier(pooled)    # (N,num_classes)

        seg_logits = self.seg_head(feats)         # (N,1,Hf,Wf)

        return class_logits, seg_logits


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

NUM_TRIALS = 2048


# ------------------------------------------------------------
# Utility: tolerant equality check
# ------------------------------------------------------------

def assert_allclose(a, b, atol=1e-4, rtol=1e-4, name=""):
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.abs(a - b)
        max_diff = float(diff.max())
        raise AssertionError(
            f"[{name}] max |a-b| = {max_diff}, "
            f"atol={atol}, rtol={rtol}"
        )


def random_int(low, high):
    return int(np.random.randint(low, high + 1))


# ------------------------------------------------------------
# Random model + input generator
# ------------------------------------------------------------

def make_random_multitask_params():
    """
    Random but valid configuration for the multitask net.

    Constraints:
      - H, W >= 4 so that after two MaxPool2d(2) we still have at least 1x1.
    """
    in_channels = random_int(1, 3)
    num_classes = random_int(2, 7)
    batch = random_int(1, 4)

    H = random_int(4, 32)
    W = random_int(4, 32)

    return {
        "in_channels": in_channels,
        "num_classes": num_classes,
        "batch": batch,
        "H": H,
        "W": W,
    }


def make_random_input(N, C, H, W):
    return np.random.uniform(-3.0, 3.0, size=(N, C, H, W)).astype(np.float32)


# ------------------------------------------------------------
# Parameter sync helpers
# ------------------------------------------------------------

def sync_scorch_to_torch(s_model: ScorchSmallConvNetMultiTask,
                         t_model: SmallConvNetMultiTask):
    """
    Copy all weights & biases from Scorch model → PyTorch model.

    Order:
      features: conv1, conv2
      classifier: fc1, fc2
      seg_head: seg_conv1, seg_conv2
    """
    # Features
    conv1_s = s_model.features[0]
    conv2_s = s_model.features[3]

    conv1_t = t_model.features[0]
    conv2_t = t_model.features[3]

    with torch.no_grad():
        conv1_t.weight[:] = torch.from_numpy(conv1_s.W)
        conv1_t.bias[:]   = torch.from_numpy(conv1_s.b)

        conv2_t.weight[:] = torch.from_numpy(conv2_s.W)
        conv2_t.bias[:]   = torch.from_numpy(conv2_s.b)

    # Classifier
    fc1_s = s_model.classifier[0]
    fc2_s = s_model.classifier[2]

    # classifier = Flatten, Linear, ReLU, Linear
    fc1_t = t_model.classifier[1]   # first Linear
    fc2_t = t_model.classifier[3]   # second Linear

    with torch.no_grad():
        fc1_t.weight[:] = torch.from_numpy(fc1_s.W)
        fc1_t.bias[:]   = torch.from_numpy(fc1_s.b)

        fc2_t.weight[:] = torch.from_numpy(fc2_s.W)
        fc2_t.bias[:]   = torch.from_numpy(fc2_s.b)

    # Segmentation head
    seg_conv1_s = s_model.seg_head[0]
    seg_conv2_s = s_model.seg_head[2]

    seg_conv1_t = t_model.seg_head[0]
    seg_conv2_t = t_model.seg_head[2]

    with torch.no_grad():
        seg_conv1_t.weight[:] = torch.from_numpy(seg_conv1_s.W)
        seg_conv1_t.bias[:]   = torch.from_numpy(seg_conv1_s.b)

        seg_conv2_t.weight[:] = torch.from_numpy(seg_conv2_s.W)
        seg_conv2_t.bias[:]   = torch.from_numpy(seg_conv2_s.b)


def collect_scorch_grads(s_model: ScorchSmallConvNetMultiTask):
    """
    Collect grad_W / grad_b arrays from all Scorch layers
    in the same logical order as sync_scorch_to_torch.
    """
    grads = []

    # Features
    conv1_s = s_model.features[0]
    conv2_s = s_model.features[3]

    grads.append(conv1_s.grad_W)
    grads.append(conv1_s.grad_b)

    grads.append(conv2_s.grad_W)
    grads.append(conv2_s.grad_b)

    # Classifier
    fc1_s = s_model.classifier[0]
    fc2_s = s_model.classifier[2]

    grads.append(fc1_s.grad_W)
    grads.append(fc1_s.grad_b)

    grads.append(fc2_s.grad_W)
    grads.append(fc2_s.grad_b)

    # Seg head
    seg_conv1_s = s_model.seg_head[0]
    seg_conv2_s = s_model.seg_head[2]

    grads.append(seg_conv1_s.grad_W)
    grads.append(seg_conv1_s.grad_b)

    grads.append(seg_conv2_s.grad_W)
    grads.append(seg_conv2_s.grad_b)

    return grads


def collect_torch_grads(t_model: SmallConvNetMultiTask):
    """
    Collect grad tensors from PyTorch layers in same order.
    """
    grads = []

    conv1_t = t_model.features[0]
    conv2_t = t_model.features[3]

    grads.append(conv1_t.weight.grad.detach().numpy())
    grads.append(conv1_t.bias.grad.detach().numpy())

    grads.append(conv2_t.weight.grad.detach().numpy())
    grads.append(conv2_t.bias.grad.detach().numpy())

    fc1_t = t_model.classifier[1]
    fc2_t = t_model.classifier[3]

    grads.append(fc1_t.weight.grad.detach().numpy())
    grads.append(fc1_t.bias.grad.detach().numpy())

    grads.append(fc2_t.weight.grad.detach().numpy())
    grads.append(fc2_t.bias.grad.detach().numpy())

    seg_conv1_t = t_model.seg_head[0]
    seg_conv2_t = t_model.seg_head[2]

    grads.append(seg_conv1_t.weight.grad.detach().numpy())
    grads.append(seg_conv1_t.bias.grad.detach().numpy())

    grads.append(seg_conv2_t.weight.grad.detach().numpy())
    grads.append(seg_conv2_t.bias.grad.detach().numpy())

    return grads


# ------------------------------------------------------------
# Core test
# ------------------------------------------------------------

def cross_verify_once(trial_index: int):
    p = make_random_multitask_params()

    in_ch      = p["in_channels"]
    num_cls    = p["num_classes"]
    N          = p["batch"]
    H          = p["H"]
    W          = p["W"]

    # Create models
    model_scorch = ScorchSmallConvNetMultiTask(
        in_channels=in_ch,
        num_classes=num_cls,
    )

    model_torch = SmallConvNetMultiTask(
        in_channels=in_ch,
        num_classes=num_cls,
    )

    # Synchronize weights
    sync_scorch_to_torch(model_scorch, model_torch)

    # Random input
    x_np = make_random_input(N, in_ch, H, W)
    x_torch = torch.from_numpy(x_np.copy()).requires_grad_(True)

    # Forward
    class_s, seg_s = model_scorch.forward(x_np)
    class_t, seg_t = model_torch(x_torch)

    class_t_np = class_t.detach().numpy()
    seg_t_np   = seg_t.detach().numpy()

    # Check forward outputs
    assert_allclose(class_s, class_t_np, name="class_logits scorch vs torch")
    assert_allclose(seg_s,   seg_t_np,   name="seg_logits scorch vs torch")

    # --------------------------------------------------------
    # Backward:
    #   We'll generate random grads for both outputs and
    #   create a fake scalar loss in PyTorch that induces
    #   exactly those grads.
    # --------------------------------------------------------
    grad_class = np.random.randn(*class_s.shape).astype(np.float32)
    grad_seg   = np.random.randn(*seg_s.shape).astype(np.float32)

    # Scorch backward
    model_scorch.zero_grad()
    grad_x_s = model_scorch.backward((grad_class.copy(), grad_seg.copy()))

    # Torch backward
    model_torch.zero_grad()
    if x_torch.grad is not None:
        x_torch.grad.zero_()

    grad_class_t = torch.from_numpy(grad_class.copy())
    grad_seg_t   = torch.from_numpy(grad_seg.copy())

    # loss = sum(class_logits * grad_class) + sum(seg_logits * grad_seg)
    loss = (class_t * grad_class_t).sum() + (seg_t * grad_seg_t).sum()
    loss.backward()

    grad_x_t = x_torch.grad.detach().numpy()

    # Compare grad wrt input
    assert_allclose(grad_x_s, grad_x_t, name="grad_x scorch vs torch")

    # Compare parameter grads
    g_s_list = collect_scorch_grads(model_scorch)
    g_t_list = collect_torch_grads(model_torch)

    if len(g_s_list) != len(g_t_list):
        raise AssertionError(
            f"[param_grads] mismatch in grad list lengths: "
            f"Scorch={len(g_s_list)}, Torch={len(g_t_list)}"
        )

    for idx, (gs, gt) in enumerate(zip(g_s_list, g_t_list)):
        assert_allclose(gs, gt, name=f"param_grad[{idx}] scorch vs torch")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(1337)
    torch.manual_seed(1337)

    print(f"[multitask_cross_verify] Running {NUM_TRIALS} random trials...\n")

    try:
        for i in range(NUM_TRIALS):
            cross_verify_once(i)
            print(f"  [OK] trial {i+1}/{NUM_TRIALS}")
    except AssertionError as e:
        print(f"\n[FAILED] trial {i}: {e}")
        raise
    else:
        print("\n[multitask_cross_verify] All multitask CNN tests passed!")
