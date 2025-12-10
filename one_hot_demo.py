# one_hot_demo.py
from __future__ import annotations
import numpy as np

# 1) Suppose these are discovered (from training CSV, in first-seen order)
demeanor_vocab = ["Brooding", "Dopey", "Gregarious", "Meticulous", "Sensual", "Sophisticated"]
emotion_vocab  = ["Daring", "Happy", "Nervous", "Provocative", "Satisfied", "Tense"]

def one_hot(value: str, vocab: list[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    idx = vocab.index(value)
    vec[idx] = 1.0
    return vec

def one_hot_decode(vec: np.ndarray, vocab: list[str]) -> str:
    idx = int(np.argmax(vec))
    return vocab[idx]

# 2) Encode a single row (demeanor+emotion → features)
demeanor = "Dopey"
emotion  = "Happy"

d_vec = one_hot(demeanor, demeanor_vocab)   # [0,1,0,0,0,0]
e_vec = one_hot(emotion,  emotion_vocab)    # [0,1,0,0,0,0]

X = np.concatenate([d_vec, e_vec])          # shape (12,)
print("Feature vector (demeanor+emotion):", X.tolist())

# 3) Decode back (to prove it’s reversible)
d_back = one_hot_decode(X[:len(demeanor_vocab)], demeanor_vocab)
e_back = one_hot_decode(X[len(demeanor_vocab):], emotion_vocab)
print("Decoded:", d_back, "/", e_back)

# 4) Target encoding (color → integer id)
color_vocab = ["Blue", "Brown", "Green", "Orange", "Red", "Yellow"]
color = "Blue"
y = color_vocab.index(color)  # integer target 0..5
print("Target class id for color:", y)
