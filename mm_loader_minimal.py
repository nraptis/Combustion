# mm_loader_minimal.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from mm_lut import MM_LUT  # uses fixed vocab + indices

def transform(df: pd.DataFrame):
    """Turn a dataframe into X (one-hot features) and y (class indices)."""
    X_rows, y_rows = [], []

    for _, row in df.iterrows():
        demeanor = row["demeanor"]
        emotion  = row["emotion"]
        color    = row["color"]

        d_vec = MM_LUT.one_hot(MM_LUT.demeanor_index(demeanor))  # 6 dims
        e_vec = MM_LUT.one_hot(MM_LUT.emotion_index(emotion))    # 6 dims
        x = np.concatenate([d_vec, e_vec]).astype(np.float32)    # 12 dims
        X_rows.append(x)

        y_rows.append(MM_LUT.color_index(color))                 # 0..5

    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=np.int64)
    return X, y

def load_mm(train_csv="mm_data_training.csv", test_csv="mm_data_testing.csv"):
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)
    Xtr, ytr = transform(df_tr)
    Xte, yte = transform(df_te)
    return (Xtr, ytr, Xte, yte)

if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_mm()
    print("Xtr shape:", Xtr.shape, "ytr shape:", ytr.shape)
    print("First X row:", Xtr[0])
    print("First y row (class idx):", ytr[0])
    # quick sanity checks
    assert Xtr.shape[1] == 12 and Xte.shape[1] == 12
    assert ytr.min() >= 0 and ytr.max() <= 5
    print("✅✅✅ loader looks good ✅✅✅")
