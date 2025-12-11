# mm_minipipeline_tf.py
from __future__ import annotations

import numpy as np
from mm_loader_minimal import load_mm
from tensorflow import Tensor
import tensorflow as tf

def main() -> None:
    Xtr, ytr, Xte, yte = load_mm()
    
    # Quick label distribution sanity check
    num_classes = 6
    train_counts = np.bincount(ytr, minlength=num_classes)
    test_counts  = np.bincount(yte, minlength=num_classes)

    print("\nTrain label counts (0..5):", train_counts)
    print("Test  label counts (0..5):", test_counts)

    # Peek at first few rows
    print("\nFirst training sample X[0]:", Xtr[0])
    print("First training label y[0]:", ytr[0])

    tensor_features = tf.convert_to_tensor(Xtr.astype(np.float32), dtype=tf.float32)
    labels = tf.convert_to_tensor(ytr.astype(np.int32), dtype=tf.int32)
    
    print("Feature Tensor Shape [TF]: ", tensor_features.shape)
    print("Label Shape [TF]", labels.shape)


if __name__ == "__main__":
    main()
