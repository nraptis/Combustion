# mm_minipipeline_torch.py
from __future__ import annotations

import numpy as np
from mm_loader_minimal import load_mm
from torch import nn
from torch import tensor

def main() -> None:
    Xtr, ytr, Xte, yte = load_mm()
    
    num_classes = 6
    train_counts = np.bincount(ytr, minlength=num_classes)
    test_counts  = np.bincount(yte, minlength=num_classes)

    print("\nTrain label counts (0..5):", train_counts)
    print("Test  label counts (0..5):", test_counts)

    print("\nFirst training sample X[0]:", Xtr[0])
    print("First training label y[0]:", ytr[0])

    tensor_features = tensor(Xtr.astype(np.float32))
    labels = ytr.astype(np.int64)

    print("Feature Tensor Shape [PyTorch]: ", tensor_features.shape)
    print("Label Shape [PyTorch]", labels.shape)


if __name__ == "__main__":
    main()
