# mm_minipipeline_torch.py
from __future__ import annotations

import numpy as np
from mm_loader_minimal import load_mm
from torch import nn, tensor
import torch


def main() -> None:
    Xtr, ytr, Xte, yte = load_mm()

    num_classes = 6
    train_counts = np.bincount(ytr, minlength=num_classes)
    test_counts  = np.bincount(yte, minlength=num_classes)

    print("\nTrain label counts (0..5):", train_counts)
    print("Test  label counts (0..5):", test_counts)

    print("\nFirst training sample X[0]:", Xtr[0])
    print("First training label y[0]:", ytr[0])

    # Convert to tensors
    tensor_features = tensor(Xtr.astype(np.float32))      # shape [N, 12]
    labels_t        = torch.from_numpy(ytr.astype(np.int64))  # shape [N]

    print("Feature Tensor Shape [PyTorch]: ", tensor_features.shape)
    print("Label Shape [PyTorch]", labels_t.shape)

    # ---- Single linear layer: 12 -> 6 ----
    in_features  = tensor_features.shape[1]  # 12
    out_features = num_classes               # 6

    model = nn.Linear(in_features=in_features,
                      out_features=out_features,
                      bias=True)

    # Forward pass (logits)
    logits = model(tensor_features)  # [N, 6]
    print("\nLogits shape [PyTorch]:", logits.shape)
    print("First logits row:", logits[0])

    # Convert logits to probabilities with softmax
    probs = torch.softmax(logits, dim=1)  # [N, 6]
    print("First probability row:", probs[0])

    # Argmax predictions
    preds = torch.argmax(probs, dim=1)  # [N]
    print("\nFirst 10 preds:", preds[:10])
    print("First 10 true labels:", labels_t[:10])

if __name__ == "__main__":
    main()
