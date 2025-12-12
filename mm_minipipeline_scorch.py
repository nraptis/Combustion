# mm_minipipeline_scorch.py
from __future__ import annotations

import numpy as np
from mm_loader_minimal import load_mm
from scorch.scorch_linear import ScorchLinear

def main() -> None:
    Xtr, ytr, Xte, yte = load_mm()

    num_classes = 6
    train_counts = np.bincount(ytr, minlength=num_classes)
    test_counts  = np.bincount(yte, minlength=num_classes)

    print("\nTrain label counts (0..5):", train_counts)
    print("Test  label counts (0..5):", test_counts)

    print("\nFirst training sample X[0]:", Xtr[0])
    print("First training label y[0]:", ytr[0])

    # Use raw NumPy features, like Torch/TF
    features = Xtr.astype(np.float32)   # shape (N, 12)
    labels   = ytr.astype(np.int64)     # shape (N,)

    print("Feature array shape [Scorch]:", features.shape)
    print("Label shape [Scorch]:", labels.shape)

    in_features  = features.shape[1]    # 12
    out_features = num_classes          # 6

    print("in_features =", in_features)
    print("out_features =", out_features)

    model = ScorchLinear(in_features=in_features,
                         out_features=out_features)

    # Forward pass (logits)
    logits = model.forward(features)    # shape (N, 6)
    print("\nLogits shape [Scorch]:", logits.shape)
    print("First logits row:", logits[0])

if __name__ == "__main__":
    main()
