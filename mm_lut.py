# mm_lut.py
from __future__ import annotations

class MM_LUT:
    # Fixed vocab orders (index 0..5)
    COLOR_VOCAB    = ["blue", "brown", "green", "orange", "red", "yellow"]
    DEMEANOR_VOCAB = ["dopey", "brooding", "sensual", "gregarious", "sophisticated", "meticulous"]
    EMOTION_VOCAB  = ["satisfied", "daring", "provocative", "happy", "tense", "nervous"]

    # Fast lookup maps
    _C2I = {v: i for i, v in enumerate(COLOR_VOCAB)}
    _D2I = {v: i for i, v in enumerate(DEMEANOR_VOCAB)}
    _E2I = {v: i for i, v in enumerate(EMOTION_VOCAB)}

    @staticmethod
    def color_index(color: str) -> int:
        return MM_LUT._C2I.get(str(color).strip().lower(), 0)  # default to blue(0)

    @staticmethod
    def demeanor_index(demeanor: str) -> int:
        return MM_LUT._D2I.get(str(demeanor).strip().lower(), 0)  # default to dopey(0)

    @staticmethod
    def emotion_index(emotion: str) -> int:
        return MM_LUT._E2I.get(str(emotion).strip().lower(), 0)  # default to satisfied(0)

    @staticmethod
    def one_hot(idx: int, length: int = 6):
        import numpy as np
        v = np.zeros(length, dtype=np.float32)
        if 0 <= idx < length:
            v[idx] = 1.0
        return v

    @staticmethod
    def decode_index(idx: int, vocab: list[str]) -> str:
        if 0 <= idx < len(vocab):
            return vocab[idx]
        return vocab[0]
