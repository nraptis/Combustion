# penguin_train_scorch.py
from __future__ import annotations

import math
from typing import Tuple, List

import numpy as np

from penguin_loader import load_penguin_dataset, PenguinDataset

from scorch.scorch_sequential import ScorchSequential
from scorch.scorch_linear import ScorchLinear
from scorch.scorch_relu import ScorchReLU
from scorch.torch_adam import TorchAdam
from scorch.scorch_optimizer import TorchParam



# ------------------------------------------------------------
# Model definition (Scorch MLP)
# ------------------------------------------------------------

class ScorchPenguinNet:
    """
    Simple MLP for tabular penguin features -> species logits,
    built entirely from Scorch modules.

        Input (D)
          -> Linear(D -> 32)
          -> ReLU
          -> Linear(32 -> 32)
          -> ReLU
          -> Linear(32 -> num_classes)
          -> logits
    """

    def __init__(self, in_features: int, num_classes: int):
        self.model = ScorchSequential(
            ScorchLinear(in_features, 32, name="fc1"),
            ScorchReLU(name="relu1"),
            ScorchLinear(32, 32, name="fc2"),
            ScorchReLU(name="relu2"),
            ScorchLinear(32, num_classes, name="fc3"),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (N, D) float32
        returns: logits (N, C)
        """
        return self.model.forward(x)

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        """
        grad_logits: dL/d(logits), shape (N, C)
        returns: dL/dx, shape (N, D)
        """
        return self.model.backward(grad_logits)

    def zero_grad(self) -> None:
        self.model.zero_grad()

    @property
    def layers(self) -> List:
        # Expose layers so we can walk the Linear layers for optimization
        return self.model.layers


# ------------------------------------------------------------
# Kaiming init for this specific MLP
# ------------------------------------------------------------

def kaiming_init_for_penguin_net(net: ScorchPenguinNet) -> None:
    """
    Reinitialize all ScorchLinear layers with Kaiming-uniform-ish weights,
    like PyTorch does for ReLU nets.

    For each Linear:
        W ~ U(-bound, +bound), where bound = sqrt(6 / fan_in)
        b = 0
    """
    for layer in net.layers:
        if isinstance(layer, ScorchLinear):
            fan_in = layer.in_features
            bound = math.sqrt(6.0 / float(fan_in))
            layer.W[...] = np.random.uniform(
                -bound, +bound, size=layer.W.shape
            ).astype(np.float32)
            layer.b[...] = 0.0


# ------------------------------------------------------------
# Loss + gradient: softmax + cross-entropy
# ------------------------------------------------------------

def softmax_cross_entropy_with_grad(
    logits: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute mean cross-entropy loss and gradient w.r.t logits.

    logits: (N, C) float32
    targets: (N,) int64 class indices in [0, C)

    Returns:
        loss: float (mean over batch)
        grad_logits: (N, C) float32, dL/d(logits)
    """
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)

    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape (N,C), got {logits.shape}")

    N, C = logits.shape
    if targets.shape != (N,):
        raise ValueError(
            f"targets shape {targets.shape} incompatible with logits shape {logits.shape}"
        )

    # Numerical stability: subtract max per row
    shifted = logits - np.max(logits, axis=1, keepdims=True)  # (N,C)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)          # (N,C)

    # Cross-entropy: mean over batch
    eps = 1e-12
    p_correct = probs[np.arange(N), targets] + eps
    losses = -np.log(p_correct)
    loss = float(losses.mean())

    # Gradient of mean CE wrt logits:
    # grad_i = (probs_i - one_hot(target_i)) / N
    grad_logits = probs.copy()
    grad_logits[np.arange(N), targets] -= 1.0
    grad_logits /= float(N)

    return loss, grad_logits.astype(np.float32)


# ------------------------------------------------------------
# Data batching helpers
# ------------------------------------------------------------

def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.RandomState,
):
    """
    Simple NumPy minibatch generator.
    """
    N = x.shape[0]
    indices = np.arange(N)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        if start >= end:
            break
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]


# ------------------------------------------------------------
# Evaluation (no gradient)
# ------------------------------------------------------------

def evaluate_model(
    net: ScorchPenguinNet,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
) -> Tuple[float, float]:
    """
    Evaluate mean loss and accuracy on a (possibly large) array.

    Returns:
        (mean_loss, accuracy)
    """
    rng = np.random.RandomState(0)  # deterministic batch order for eval
    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in iterate_minibatches(x, y, batch_size=batch_size, shuffle=False, rng=rng):
        logits = net.forward(xb)                # (B,C)
        loss, _ = softmax_cross_entropy_with_grad(logits, yb)
        total_loss += loss * xb.shape[0]

        preds = np.argmax(logits, axis=1)
        total_correct += int((preds == yb).sum())
        total += xb.shape[0]

    mean_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1) if total > 0 else 0.0
    return mean_loss, acc


# ------------------------------------------------------------
# Training loop (momentum SGD)
# ------------------------------------------------------------

def train_penguin_net_scorch(
    csv_path: str = "penguins_cleaned.csv",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-2,          # higher LR than the Torch Adam (since this is plain momentum SGD)
    momentum: float = 0.9,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    random_seed: int = 1337,
):
    # ------------------------------------------------------
    # Repro-ish seeds
    # ------------------------------------------------------
    rng = np.random.RandomState(random_seed)
    np.random.seed(random_seed)

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    ds = load_penguin_dataset(
        csv_path=csv_path,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    num_classes = len(ds.species_to_index)
    in_features = ds.x_train.shape[1]

    print(f"[penguin_train_scorch] Classes: {ds.species_to_index}")
    print(f"[penguin_train_scorch] Input dim: {in_features}")
    print(f"[penguin_train_scorch] Train/Val/Test sizes: "
          f"{ds.x_train.shape[0]} / {ds.x_val.shape[0]} / {ds.x_test.shape[0]}")

    # Model
    net = ScorchPenguinNet(in_features=in_features, num_classes=num_classes)

    # Use Kaiming-like init to match modern ReLU conventions
    kaiming_init_for_penguin_net(net)

    # ------------------------------------------------------
    # Momentum SGD state: one velocity per Linear param
    # ------------------------------------------------------
    velocity = []  # list of dicts per linear layer

    for layer in net.layers:
        if isinstance(layer, ScorchLinear):
            v_W = np.zeros_like(layer.W, dtype=np.float32)
            v_b = np.zeros_like(layer.b, dtype=np.float32)
            velocity.append({
                "layer": layer,
                "v_W": v_W,
                "v_b": v_b,
            })

    params: list[TorchParam] = []
    for layer in net.layers:
        if isinstance(layer, ScorchLinear):
            params.append(TorchParam(data=layer.W, grad=layer.grad_W))
            params.append(TorchParam(data=layer.b, grad=layer.grad_b))

    optimizer = TorchAdam(params, lr=1e-3)

    # ------------------------------------------------------
    # Training loop
    # ------------------------------------------------------
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in iterate_minibatches(
            ds.x_train, ds.y_train,
            batch_size=batch_size,
            shuffle=True,
            rng=rng,
        ):
            # Forward
            logits = net.forward(xb)  # (B, C)

            # Loss + grad wrt logits
            loss, grad_logits = softmax_cross_entropy_with_grad(logits, yb)

            # Bookkeeping for stats
            running_loss += loss * xb.shape[0]
            preds = np.argmax(logits, axis=1)
            running_correct += int((preds == yb).sum())
            running_total += xb.shape[0]

            # Backward through network
            net.zero_grad()
            _ = net.backward(grad_logits)
            optimizer.step()

            # Momentum SGD step
            for entry in velocity:
                layer = entry["layer"]
                v_W = entry["v_W"]
                v_b = entry["v_b"]

                # v <- mu * v - lr * grad
                v_W[:] = momentum * v_W - lr * layer.grad_W
                v_b[:] = momentum * v_b - lr * layer.grad_b

                # w <- w + v
                layer.W += v_W
                layer.b += v_b

        # Epoch-level stats
        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1) if running_total > 0 else 0.0

        val_loss, val_acc = evaluate_model(
            net, ds.x_val, ds.y_val, batch_size=256
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

    # ------------------------------------------------------
    # Final test eval
    # ------------------------------------------------------
    test_loss, test_acc = evaluate_model(
        net, ds.x_test, ds.y_test, batch_size=256
    )
    print(
        f"[penguin_train_scorch] TEST: "
        f"loss={test_loss:.4f}, acc={test_acc:.3f}"
    )

    return net, ds


if __name__ == "__main__":
    # Assumes penguins_cleaned.csv is in the same directory
    train_penguin_net_scorch()
