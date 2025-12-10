# mm_loader_minimal.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def build_vocab(values: list[str]) -> list[str]:
    """Stable vocab in order of first appearance."""
    seen = set()
    for v in values:
        seen.add(v)
    return list(seen)

def one_hot(value: str, vocab: list[str]) -> np.ndarray:
    k = len(vocab)
    vec = np.zeros(k, dtype=np.float32)
    try:
        idx = vocab.index(value)
    except ValueError:
        # if value not found, we expect vocab to include "UNK"
        idx = vocab.index("UNK")
    vec[idx] = 1.0
    return vec

def fit_vocabs(train_csv: str | Path):
    """Scan the training CSV to make vocabs for color (target), demeanor, emotion."""
    df = pd.read_csv(train_csv)
    color_vocab   = build_vocab(df["color"].tolist())
    demeanor_vocab= build_vocab(df["demeanor"].tolist())
    emotion_vocab = build_vocab(df["emotion"].tolist())
    return {
        "color": color_vocab,
        "demeanor": demeanor_vocab + (["UNK"] if "UNK" not in demeanor_vocab else []),
        "emotion":  emotion_vocab + (["UNK"] if "UNK" not in emotion_vocab  else []),
    }

def transform(df: pd.DataFrame, vocabs: dict[str, list[str]]):
    """Turn a dataframe into X (one-hot features) and y (class indices) using given vocabs."""
    X_rows = []
    y_rows = []

    for _, row in df.iterrows():
        d_vec = one_hot(row["demeanor"], vocabs["demeanor"])   # length = len(demeanor_vocab)
        e_vec = one_hot(row["emotion"],  vocabs["emotion"])    # length = len(emotion_vocab)
        x = np.concatenate([d_vec, e_vec]).astype(np.float32)  # final feature vector
        X_rows.append(x)

        # target is the index of color in color vocab
        color_idx = vocabs["color"].index(row["color"])
        y_rows.append(color_idx)

    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=np.int64)
    return X, y

def load_mm(train_csv="mm_data_training.csv", test_csv="mm_data_testing.csv"):
    vocabs = fit_vocabs(train_csv)
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    Xtr, ytr = transform(df_tr, vocabs)
    Xte, yte = transform(df_te, vocabs)

    return (Xtr, ytr, Xte, yte), vocabs

if __name__ == "__main__":
    (Xtr, ytr, Xte, yte), vocabs = load_mm()
    print("Feature dims:",
          len(vocabs["demeanor"]), "+", len(vocabs["emotion"]),
          "=", len(vocabs["demeanor"]) + len(vocabs["emotion"]))
    print("Xtr shape:", Xtr.shape, "ytr shape:", ytr.shape)
    print("First X row:", Xtr[0])
    print("First y row (class idx):", ytr[0])
    print("Color vocab:", vocabs["color"])
    print("Demeanor vocab:", vocabs["demeanor"])
    print("Emotion vocab:", vocabs["emotion"])
