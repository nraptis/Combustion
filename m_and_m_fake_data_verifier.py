# m_and_m_fake_data_verifier.py
# Verify synthetic M&M data by plotting histograms of demeanor/emotion per color.

from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Canonical label sets
# -----------------------------
COLORS = ["Blue", "Brown", "Green", "Orange", "Red", "Yellow"]

PRIMARY_DEMEANOR = {
    "Blue": "Dopey",
    "Brown": "Brooding",
    "Green": "Sensual",
    "Orange": "Gregarious",
    "Red": "Sophisticated",
    "Yellow": "Meticulous",
}
# Gather all known demeanor labels from the mapping (keeps order stable)
ALL_DEMEANORS = list(dict.fromkeys(PRIMARY_DEMEANOR.values()))

PRIMARY_EMOTION = {
    "Blue": "Satisfied",
    "Brown": "Daring",
    "Green": "Provocative",
    "Orange": "Happy",
    "Red": "Tense",
    "Yellow": "Nervous",
}
ALL_EMOTIONS = list(dict.fromkeys(PRIMARY_EMOTION.values()))

# -----------------------------
# Utilities
# -----------------------------
def read_data(train_csv: Path, test_csv: Path, which: str) -> pd.DataFrame:
    if which not in {"train", "test", "both"}:
        raise ValueError("--which must be one of: train, test, both")

    if which == "train":
        df = pd.read_csv(train_csv)
        df["split"] = "train"
        return df
    if which == "test":
        df = pd.read_csv(test_csv)
        df["split"] = "test"
        return df

    # both
    df_tr = pd.read_csv(train_csv)
    df_tr["split"] = "train"
    df_te = pd.read_csv(test_csv)
    df_te["split"] = "test"
    return pd.concat([df_tr, df_te], ignore_index=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_bar_counts(
    counts: pd.Series,
    title: str,
    outfile: Path,
    ylabel: str = "Count",
    xtick_rotation: int = 20,
    annotate_pct: bool = True,
):
    plt.figure(figsize=(7, 5))
    ax = counts.plot(kind="bar")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=xtick_rotation)

    if annotate_pct:
        total = counts.sum()
        for p in ax.patches:
            height = p.get_height()
            if total > 0:
                pct = 100.0 * (height / total)
                ax.annotate(
                    f"{int(height)} ({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def per_color_histograms(
    df: pd.DataFrame,
    outdir: Path,
    include_tables: bool = True,
):
    ensure_dir(outdir)

    # Ensure category order so bars appear in a stable order even if some values are missing
    df = df.copy()
    df["color"] = pd.Categorical(df["color"], categories=COLORS, ordered=True)
    df["demeanor"] = pd.Categorical(df["demeanor"], categories=ALL_DEMEANORS, ordered=True)
    df["emotion"] = pd.Categorical(df["emotion"], categories=ALL_EMOTIONS, ordered=True)

    for color in COLORS:
        sub = df[df["color"] == color]

        # Demeanor histogram for this color
        dem_counts = sub["demeanor"].value_counts().reindex(ALL_DEMEANORS, fill_value=0)
        dem_title = f"Demeanor distribution for {color}"
        dem_file = outdir / f"hist_demeanor_by_color_{color}.png"
        plot_bar_counts(dem_counts, dem_title, dem_file)

        # Emotion histogram for this color
        emo_counts = sub["emotion"].value_counts().reindex(ALL_EMOTIONS, fill_value=0)
        emo_title = f"Emotion distribution for {color}"
        emo_file = outdir / f"hist_emotion_by_color_{color}.png"
        plot_bar_counts(emo_counts, emo_title, emo_file)

        if include_tables:
            total_n = len(sub)
            dem_pct = (dem_counts / total_n * 100.0).round(1).astype(str) + "%"
            emo_pct = (emo_counts / total_n * 100.0).round(1).astype(str) + "%"
            print(f"\n=== {color}: Demeanor counts ===")
            print(pd.DataFrame({"count": dem_counts, "percent": dem_pct}))
            print(f"\n=== {color}: Emotion counts ===")
            print(pd.DataFrame({"count": emo_counts, "percent": emo_pct}))


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate per-color histograms for demeanor and emotion."
    )
    ap.add_argument(
        "--train",
        type=Path,
        default=Path("mm_data_training.csv"),
        help="Path to training CSV (default: mm_data_training.csv)",
    )
    ap.add_argument(
        "--test",
        type=Path,
        default=Path("mm_data_testing.csv"),
        help="Path to testing CSV (default: mm_data_testing.csv)",
    )
    ap.add_argument(
        "--which",
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to visualize (default: both, concatenated)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("mm_plots"),
        help="Directory to write PNGs (default: mm_plots/)",
    )
    ap.add_argument(
        "--no-tables",
        action="store_true",
        help="Do not print count/percentage tables to stdout.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    df = read_data(args.train, args.test, which=args.which)
    per_color_histograms(
        df=df,
        outdir=args.outdir,
        include_tables=not args.no_tables,
    )
    print(f"\nWrote plots to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
