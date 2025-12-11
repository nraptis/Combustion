# runner_scorch_new.py
from __future__ import annotations

import numpy as np

from scorch.scorch_linear import ScorchLinear
from scorch.scorch_relu import ScorchReLU
from scorch.scorch_sequential import ScorchSequential
from scorch_ext.data_loader import DataLoader
from scorch_ext.tensor_load_helpers import image_and_mask_from_pixel_bag
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
# Training helper: one SGD step on all ScorchLinear layers
# ------------------------------------------------------
def sgd_step(model: ScorchSequential, lr: float):
    """
    Simple SGD update:
      param = param - lr * grad

    Only updates layers that have W / b / grad_W / grad_b
    (i.e., ScorchLinear). ReLU layers are skipped because
    they have no parameters.
    """
    for layer in model.layers:
        if hasattr(layer, "W") and hasattr(layer, "grad_W"):
            layer.W -= lr * layer.grad_W
        if hasattr(layer, "b") and hasattr(layer, "grad_b"):
            layer.b -= lr * layer.grad_b


# ------------------------------------------------------
# Build dataset from your patch loader
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

    # You said you added this:
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
                # Uncomment if you want to see skipped shapes:
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
# Main runner
# ------------------------------------------------------
def runner_scorch_new():
    print("=== SCORCH NEW TRAINER START ===")

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

    # 2) Build a small model: Linear → ReLU → Linear
    hidden_dim = 16  # 16 or 64 also fine

    model = ScorchSequential(
        
        ScorchLinear(input_dim, hidden_dim, name="Linear_0"),
        
        ScorchReLU(name="ReLU_0"),
        
        ScorchLinear(hidden_dim, num_classes, name="Linear_1"),
    )

    print("[Model]")
    print(model)

    # 3) Training hyperparameters
    epochs = 200
    learning_rate = 0.01

    # 4) Training loop (per-sample SGD)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0

        # Shuffle the sample order each epoch (optional but nice)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for idx in indices:
            x_sample = X[idx]          # shape (input_dim,)
            y_sample = int(y[idx])     # scalar label

            # Clear gradients
            model.zero_grad()

            # ---- Forward ----
            logits = model.forward(x_sample)   # (num_classes,)

            # Prediction for accuracy
            pred_class = int(np.argmax(logits))
            if pred_class == y_sample:
                correct += 1

            # ---- Loss + gradient wrt logits ----
            loss, grad_logits = softmax_and_cross_entropy_with_grad(logits, y_sample)
            total_loss += loss

            # ---- Backward ----
            _ = model.backward(grad_logits)

            # ---- SGD parameter update ----
            sgd_step(model, lr=learning_rate)

        avg_loss = total_loss / num_samples
        accuracy = correct / num_samples

        print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f}  acc={accuracy:.3f}")

    print("=== SCORCH NEW TRAINER DONE ===")

    # 5) Show a few example predictions at the end
    print("\n[Samples after training]")
    for i in range(min(5, num_samples)):
        x_sample = X[i]
        y_true = int(y[i])
        logits = model.forward(x_sample)
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)
        y_pred = int(np.argmax(probs))
        print(
            f"  sample={i:02d}, true={class_names[y_true]}, pred={class_names[y_pred]}, "
            f"probs={np.round(probs, 3)}"
        )


if __name__ == "__main__":
    runner_scorch_new()
