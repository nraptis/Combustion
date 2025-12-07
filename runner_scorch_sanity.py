# runner_scorch_sanity.py
from __future__ import annotations

import numpy as np

from scorch.scorch_sequential import ScorchSequential
from scorch.scorch_linear_2 import ScorchLinear2
from scorch.scorch_relu_2 import ScorchReLU2

from scorch.data_loader import DataLoader
from scorch.tensor_load_helpers import image_and_mask_from_pixel_bag
from image.bitmap import Bitmap


# ------------------------------------------------------
# Softmax + cross-entropy with gradient for a single sample
# ------------------------------------------------------
def softmax_and_cross_entropy_with_grad(logits: np.ndarray, target_index: int):
    """
    Compute softmax probabilities, cross-entropy loss, and
    the gradient of the loss with respect to the logits.

    logits: (C,) - raw scores from the network
    target_index: int in [0, C)

    Returns:
        loss: float
        grad_logits: np.ndarray, same shape as logits
    """
    # Numerical stability: shift by max
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)

    # Cross-entropy loss: -log(prob_of_correct_class)
    loss = -np.log(probs[target_index] + 1e-12)

    # Gradient of loss w.r.t. logits:
    #   grad = probs
    #   grad[target] -= 1
    grad = probs.copy()
    grad[target_index] -= 1.0

    return float(loss), grad


# ------------------------------------------------------
# Build dataset from your patch loader (same as runner_scorch_new)
# ------------------------------------------------------
def build_patch_dataset_from_loader(
    annotations_subdir: str,
    images_subdir: str | None = None,
    target_h: int = 40,
    target_w: int = 40,
):
    """
    Walk your DataLoader, extract image patches for each label, and build
    (X, y) where:

        X: (N, D)  flattened grayscale patches
        y: (N,)    integer class indices

    Currently only keeps patches with exact size (1, target_h, target_w)
    to keep the ScorchLinear input dimension fixed.
    """
    loader = DataLoader(
        annotations_subdir=annotations_subdir,
        images_subdir=images_subdir,
    )

    class_names = loader.class_names      # e.g. ["Red", "Blue", "Green"]
    print("class names were =>", class_names)

    num_classes = len(class_names)
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    xs = []
    ys = []

    total_patches = 0
    used_patches = 0

    for pair_index, pair in enumerate(loader):
        bmp = Bitmap.with_image(pair.image_path)
        doc = pair.document

        print(f"[Dataset] Pair {pair_index}: {pair.image_path}")

        for label in doc.data.labels:
            total_patches += 1
            bag = label.pixel_bag
            true_name = label.name

            # Crop patch
            img_patch, _mask_patch = image_and_mask_from_pixel_bag(
                bmp=bmp,
                bag=bag,
                name=label.name,
                grayscale=True,
            )

            c, h, w = img_patch.shape

            # Only keep exact size (1, target_h, target_w) for now
            if c != 1 or h != target_h or w != target_w:
                # print(f"  Skipping {true_name} patch with shape {img_patch.shape}")
                continue

            used_patches += 1

            # Flatten to (D,)
            arr = img_patch.to_numpy().astype(np.float32)  # (1, H, W)
            x_flat = arr.reshape(-1)                       # (H * W,)

            class_idx = label_to_idx[true_name]

            xs.append(x_flat)
            ys.append(class_idx)

            print(f"  Using patch {true_name} with shape {img_patch.shape} -> flat {x_flat.shape}")

    if not xs:
        raise RuntimeError(
            "No patches matched the target size. "
            "Try adjusting target_h/target_w or implement padding/resizing."
        )

    X = np.stack(xs, axis=0)              # (N, D)
    y = np.array(ys, dtype=np.int64)      # (N,)

    print(f"[Dataset] total_patches={total_patches}, used_patches={used_patches}")
    print(f"[Dataset] X.shape={X.shape}, y.shape={y.shape}, num_classes={num_classes}")

    return X, y, class_names


# ------------------------------------------------------
# Gradient check helpers (for a ScorchSequential model)
# ------------------------------------------------------
def compute_loss(model: ScorchSequential, x: np.ndarray, target_index: int) -> float:
    """
    Forward through the model and return scalar loss for a single (x, y).
    """
    logits = model.forward(x)  # (C,)
    loss, _ = softmax_and_cross_entropy_with_grad(logits, target_index)
    return loss


def compute_loss_and_backprop(
    model: ScorchSequential,
    x: np.ndarray,
    target_index: int,
) -> float:
    """
    Forward + backward for a single sample.
    Fills model gradients via backward().
    """
    model.zero_grad()

    logits = model.forward(x)
    loss, grad_logits = softmax_and_cross_entropy_with_grad(logits, target_index)
    _ = model.backward(grad_logits)

    return loss


def grad_check_weight(
    model: ScorchSequential,
    x: np.ndarray,
    target_index: int,
    layer_idx: int,
    i: int,
    j: int,
    eps: float = 1e-4,
):
    """
    Compare analytic grad_W[i,j] vs numeric finite-difference gradient
    on a given ScorchLinear2 layer inside a ScorchSequential.

    layer_idx:
        index in model.layers where the ScorchLinear2 lives.

    i, j:
        indices into that layer's W matrix.
    """
    layer = model.layers[layer_idx]
    if not isinstance(layer, ScorchLinear2):
        raise TypeError(f"Layer at index {layer_idx} is not ScorchLinear2, got {type(layer)}")

    W = layer.W

    # --- 1) Analytic gradient via backprop ---
    _ = compute_loss_and_backprop(model, x, target_index)
    analytic = layer.grad_W[i, j]

    # --- 2) Numeric gradient via finite differences ---
    original = W[i, j]

    # W[i, j] + eps
    W[i, j] = original + eps
    loss_plus = compute_loss(model, x, target_index)

    # W[i, j] - eps
    W[i, j] = original - eps
    loss_minus = compute_loss(model, x, target_index)

    # Restore
    W[i, j] = original

    numeric = (loss_plus - loss_minus) / (2.0 * eps)

    diff = abs(analytic - numeric)

    print(f"[grad_check] layer={layer_idx}, W[{i},{j}]")
    print(f"  analytic: {analytic}")
    print(f"  numeric : {numeric}")
    print(f"  diff    : {diff}")

    return analytic, numeric, diff


# ------------------------------------------------------
# Main sanity runner
# ------------------------------------------------------
def runner_scorch_sanity():
    print("=== SCORCH SANITY START ===")

    # 1) Build dataset from real patches (filtered to 40x40)
    X, y, class_names = build_patch_dataset_from_loader(
        annotations_subdir="training_tiny",
        images_subdir="training_tiny",
        target_h=40,
        target_w=40,
    )

    num_samples, input_dim = X.shape
    num_classes = len(class_names)

    print(f"[Data] num_samples={num_samples}, input_dim={input_dim}, num_classes={num_classes}")
    print(f"[Data] classes = {class_names}")

    # 2) Build model: Linear2 → ReLU2 → Linear2
    hidden_dim = 32

    model = ScorchSequential(
        ScorchLinear2(input_dim, hidden_dim, name="Linear2_0"),
        ScorchReLU2(name="ReLU2_0"),
        ScorchLinear2(hidden_dim, num_classes, name="Linear2_1"),
    )

    print("[Model]")
    print(model)

    # 3) Pick one sample to test gradients on
    sample_idx = 0
    x_sample = X[sample_idx]
    y_sample = int(y[sample_idx])

    print(f"\n[Sanity] Using sample_idx={sample_idx}, true class={class_names[y_sample]}")

    # 4) Run gradient checks on a few weights in first and second linear layers
    print("\n=== Gradient check: first linear layer (layer_idx=0) ===")
    grad_check_weight(model, x_sample, y_sample, layer_idx=0, i=0, j=0)
    grad_check_weight(model, x_sample, y_sample, layer_idx=0, i=1, j=5)

    print("\n=== Gradient check: second linear layer (layer_idx=2) ===")
    grad_check_weight(model, x_sample, y_sample, layer_idx=2, i=0, j=0)
    grad_check_weight(model, x_sample, y_sample, layer_idx=2, i=1, j=3)

    print("\n=== SCORCH SANITY DONE ===")


if __name__ == "__main__":
    runner_scorch_sanity()
