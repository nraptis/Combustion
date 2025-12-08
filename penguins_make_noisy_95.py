# penguins_make_noisy_95.py
from __future__ import annotations

import numpy as np
import pandas as pd


def make_noisy_penguins(
    input_csv: str = "penguins_cleaned.csv",
    output_csv: str = "penguins_noisy_95.csv",
    noise_rate: float = 0.05,
    random_seed: int = 1337,
) -> None:
    """
    Create a penguin dataset with random label noise.

    We flip `noise_rate` fraction of rows to a *different* random species.
    If noise_rate = 0.05, then even a perfect classifier can only reach
    about 95% accuracy w.r.t. these noisy labels on *unseen* data.

    This affects *only* the 'species' column.
    """
    rng = np.random.RandomState(random_seed)

    df = pd.read_csv(input_csv)
    if "species" not in df.columns:
        raise ValueError(f"'species' column not found in {input_csv}")

    species = df["species"].values
    unique_species = sorted(df["species"].unique())
    num_classes = len(unique_species)

    if num_classes < 2:
        raise ValueError("Need at least 2 species to inject meaningful label noise.")

    n = len(df)
    n_flip = int(round(noise_rate * n))

    print(f"[penguins_make_noisy_95] Loaded {n} rows from {input_csv}")
    print(f"[penguins_make_noisy_95] Unique species: {unique_species}")
    print(f"[penguins_make_noisy_95] Flipping labels for ~{n_flip} rows "
          f"({noise_rate * 100:.1f}% noise)")

    # Choose random rows to corrupt
    flip_indices = rng.choice(n, size=n_flip, replace=False)

    # For each selected row, pick a *different* random species
    for idx in flip_indices:
        current = species[idx]
        # All species except current
        other_species = [s for s in unique_species if s != current]
        new_label = rng.choice(other_species)
        species[idx] = new_label

    # Write back modified labels
    df["species"] = species

    df.to_csv(output_csv, index=False)
    print(f"[penguins_make_noisy_95] Saved noisy dataset to {output_csv}")


if __name__ == "__main__":
    make_noisy_penguins()
