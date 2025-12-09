# penguin_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS: List[str] = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]

TARGET_COLUMN = "species"

@dataclass
class PenguinDataset:
    """
    Simple container for the Penguin tabular dataset.

    All features are float32, standardized with train mean/std.
    Targets are integer class indices.
    """
    x_train: np.ndarray  # (N_train, D)
    y_train: np.ndarray  # (N_train,)
    x_val: np.ndarray    # (N_val, D)
    y_val: np.ndarray    # (N_val,)
    x_test: np.ndarray   # (N_test, D)
    y_test: np.ndarray   # (N_test,)

    feature_names: List[str]
    species_to_index: Dict[str, int]
    index_to_species: Dict[int, str]

    mean: np.ndarray      # (D,)
    std: np.ndarray       # (D,)


def _encode_sex_column(df: pd.DataFrame) -> pd.Series:
    """
    Encode 'sex' as numeric:

        female -> 0.0
        male   -> 1.0

    Any unexpected / missing values become 0.5 (neutral midpoint).
    """
    col = df["sex"].astype(str)

    mapping = {
        "female": 0.0,
        "male": 1.0,
    }

    encoded = col.map(mapping)

    # Handle unknowns / NaNs
    encoded = encoded.fillna(0.5).astype(np.float32)

    return encoded


def _encode_species_column(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Encode species as class indices 0..K-1.
    """
    species = df[TARGET_COLUMN].astype(str)
    unique_species = sorted(species.unique().tolist())

    species_to_index = {name: idx for idx, name in enumerate(unique_species)}
    index_to_species = {idx: name for name, idx in species_to_index.items()}

    y = species.map(species_to_index).astype(np.int64).to_numpy()

    return y, species_to_index, index_to_species


def _train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Simple manual split into train / val / test with shuffling.

    Fractions are relative to the full dataset.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0,1), got {val_fraction}")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0,1), got {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    N = x.shape[0]
    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(N)

    n_test = int(round(N * test_fraction))
    n_val = int(round(N * val_fraction))
    n_train = N - n_val - n_test

    if n_train <= 0:
        raise ValueError("Not enough samples left for training after splitting.")

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_penguin_dataset(
    csv_path: str | Path = "penguins_noisy_95.csv",
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    random_seed: int = 1337,
) -> PenguinDataset:
    """
    Load and preprocess the penguin dataset.

    Steps:
      - Read CSV.
      - Encode sex (0/1).
      - Encode species → class indices.
      - Standardize features using *train* mean/std.
      - Split into train/val/test.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        # Try relative to this file's directory
        alt = Path(__file__).resolve().parent / csv_path.name
        if alt.exists():
            csv_path = alt
        else:
            raise FileNotFoundError(f"Could not find CSV at {csv_path} or {alt}")

    df = pd.read_csv(csv_path)

    # Encode sex into numeric column
    df = df.copy()
    df["sex"] = _encode_sex_column(df)

    # Extract feature matrix
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # Encode target
    y, species_to_index, index_to_species = _encode_species_column(df)

    # Initial split (before normalization)
    x_train, y_train, x_val, y_val, x_test, y_test = _train_val_test_split(
        x, y,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    # Standardization: compute mean/std on train only
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)

    # Avoid div-by-zero
    std_safe = std.copy()
    std_safe[std_safe == 0.0] = 1.0

    def norm(z: np.ndarray) -> np.ndarray:
        return ((z - mean) / std_safe).astype(np.float32)

    x_train = norm(x_train)
    x_val = norm(x_val)
    x_test = norm(x_test)

    return PenguinDataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_names=FEATURE_COLUMNS,
        species_to_index=species_to_index,
        index_to_species=index_to_species,
        mean=mean.astype(np.float32),
        std=std_safe.astype(np.float32),
    )


if __name__ == "__main__":
    # Quick smoke test
    ds = load_penguin_dataset()
    print("[penguin_loader] Shapes:")
    print("  x_train:", ds.x_train.shape, "y_train:", ds.y_train.shape)
    print("  x_val:  ", ds.x_val.shape,   "y_val:  ", ds.y_val.shape)
    print("  x_test: ", ds.x_test.shape,  "y_test: ", ds.y_test.shape)
    print("  classes:", ds.species_to_index)
    print("  features:", ds.feature_names)
