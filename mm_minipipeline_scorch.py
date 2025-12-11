# mm_minipipeline_scorch.py
from __future__ import annotations

import numpy as np
from mm_loader_minimal import load_mm
from scorch.scorch_tensor import ScorchTensor

def main() -> None:
    Xtr, ytr, Xte, yte = load_mm()
    
    num_classes = 6
    train_counts = np.bincount(ytr, minlength=num_classes)
    test_counts  = np.bincount(yte, minlength=num_classes)

    print("\nTrain label counts (0..5):", train_counts)
    print("Test  label counts (0..5):", test_counts)

    print("\nFirst training sample X[0]:", Xtr[0])
    print("First training label y[0]:", ytr[0])


    tensor_features = ScorchTensor(Xtr.astype(np.float32), name="features")
    labels = ytr.astype(np.int32)

    print("Feature Tensor Shape [Scorch]: ", tensor_features.shape)
    print("Label Shape [Scorch]", labels.shape)
    


if __name__ == "__main__":
    main()
