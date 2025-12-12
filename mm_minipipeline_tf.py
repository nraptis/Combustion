# mm_minipipeline_tf.py
from __future__ import annotations

import numpy as np
from mm_loader_minimal import load_mm
import tensorflow as tf


def main() -> None:
    Xtr, ytr, Xte, yte = load_mm()

    # Quick label distribution sanity check
    num_classes = 6
    train_counts = np.bincount(ytr, minlength=num_classes)
    test_counts  = np.bincount(yte, minlength=num_classes)

    print("\nTrain label counts (0..5):", train_counts)
    print("Test  label counts (0..5):", test_counts)

    print("\nFirst training sample X[0]:", Xtr[0])
    print("First training label y[0]:", ytr[0])

    # Convert to tensors
    tensor_features = tf.convert_to_tensor(Xtr.astype(np.float32), dtype=tf.float32)  # [N, 12]
    labels = tf.convert_to_tensor(ytr.astype(np.int32), dtype=tf.int32)               # [N]

    print("Feature Tensor Shape [TF]: ", tensor_features.shape)
    print("Label Shape [TF]", labels.shape)

    # ---- Single Dense (linear) layer: 12 -> 6 ----
    in_features  = tensor_features.shape[1]  # 12
    num_classes  = 6

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, input_shape=(in_features,))
    ])

    # Forward pass (logits)
    logits = model(tensor_features)  # [N, 6]
    print("\nLogits shape [TF]:", logits.shape)
    print("First logits row:", logits[0].numpy())

    # Softmax to probabilities
    probs = tf.nn.softmax(logits, axis=1)  # [N, 6]
    print("First probability row:", probs[0].numpy())

    # Argmax predictions
    preds = tf.argmax(probs, axis=1)  # [N]
    print("\nFirst 10 preds:", preds[:10].numpy())
    print("First 10 true labels:", labels[:10].numpy())


if __name__ == "__main__":
    main()
