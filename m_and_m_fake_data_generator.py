# mm_fake_data_generator.py
# Generate synthetic M&M data with color → primary demeanor/emotion,
# plus controlled off-primary randomness.

from __future__ import annotations
import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
TRAIN_N = 1000
TEST_N  = 1000
SEED    = 1337  # change for a different draw

COLORS = ["Blue", "Brown", "Green", "Orange", "Red", "Yellow"]

PRIMARY_DEMEANOR = {
    "Blue": "Dopey",
    "Brown": "Brooding",
    "Green": "Sensual",
    "Orange": "Gregarious",
    "Red": "Sophisticated",
    "Yellow": "Meticulous",
}

PRIMARY_EMOTION = {
    "Blue": "Satisfied",
    "Brown": "Daring",
    "Green": "Provocative",
    "Orange": "Happy",
    "Red": "Tense",
    "Yellow": "Nervous",
}

P_PRIMARY_DEMEANOR = 0.90  # 90% primary, 10% any other
P_PRIMARY_EMOTION  = 0.80  # 80% primary, 20% any other

# -----------------------------
# Core sampler helpers
# -----------------------------
rng = np.random.default_rng(SEED)

def draw_with_primary(color: str, mapping: dict[str, str], p_primary: float) -> str:
    """
    With probability p_primary, return the mapped primary for this color.
    Otherwise pick uniformly from other colors' primaries.
    """
    if rng.random() < p_primary:
        return mapping[color]
    # choose uniformly among other primaries
    others = [v for k, v in mapping.items() if k != color]
    return rng.choice(others)

def make_split(n_rows: int) -> pd.DataFrame:
    colors = rng.choice(COLORS, size=n_rows)
    records = []
    for c in colors:
        demeanor = draw_with_primary(c, PRIMARY_DEMEANOR, P_PRIMARY_DEMEANOR)
        emotion  = draw_with_primary(c, PRIMARY_EMOTION,  P_PRIMARY_EMOTION)
        records.append({
            "color": c,
            "demeanor": demeanor,
            "emotion": emotion,
        })
    return pd.DataFrame.from_records(records)

# -----------------------------
# Generate and save
# -----------------------------
if __name__ == "__main__":
    train_df = make_split(TRAIN_N)
    test_df  = make_split(TEST_N)

    train_df.to_csv("mm_data_training.csv", index=False)
    test_df.to_csv("mm_data_testing.csv", index=False)

    # Quick sanity print
    print("Wrote mm_data_training.csv:", len(train_df), "rows")
    print("Wrote mm_data_testing.csv:", len(test_df), "rows")

    # Optional peek
    print("\nSample rows:")
    print(train_df.head(5))
