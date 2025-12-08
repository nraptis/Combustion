# penguin_plots_ii_noisy.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for projection="3d"


def main():
    # Load noisy penguins dataset
    df = pd.read_csv("penguins_noisy_95.csv")

    print(df.head())
    sns.set(style="whitegrid", context="talk")

    # ------------------------------------------------------------
    # 1) Pairplot of the 4 numeric features (NOISY LABELS)
    # ------------------------------------------------------------

    features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    print("[penguin_plots_ii_noisy] Generating pairplot (noisy labels)...")

    pairplot_fig = sns.pairplot(
        df,
        vars=features,
        hue="species",
        palette="deep",
        diag_kind="kde",
        plot_kws={"alpha": 0.7, "s": 60},
        height=2.4,
    )
    pairplot_fig.fig.suptitle(
        "Penguins Pairplot (noisy species labels, ~95% max possible acc)",
        y=1.02,
    )

    plt.show()

    # ------------------------------------------------------------
    # 2) 3D scatter plot (NOISY LABELS)
    # ------------------------------------------------------------
    print("[penguin_plots_ii_noisy] Generating 3D scatter (noisy labels)...")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    species_unique = df["species"].unique()
    palette = sns.color_palette("deep", n_colors=len(species_unique))

    for species_name, color in zip(species_unique, palette):
        subset = df[df["species"] == species_name]
        ax.scatter(
            subset["bill_length_mm"],
            subset["bill_depth_mm"],
            subset["flipper_length_mm"],
            s=60,
            alpha=0.8,
            label=species_name,
            color=color,
        )

    ax.set_xlabel("bill_length_mm")
    ax.set_ylabel("bill_depth_mm")
    ax.set_zlabel("flipper_length_mm")
    ax.set_title("3D Scatter (noisy labels): bill_length × bill_depth × flipper_length")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
