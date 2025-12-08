# penguin_plots.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("penguins_cleaned.csv")

    print(df.head())

    sns.set(style="whitegrid", context="talk")

    # Feature combinations to plot
    pairs = [
        ("bill_length_mm", "bill_depth_mm"),
        ("bill_length_mm", "flipper_length_mm"),
        ("bill_length_mm", "body_mass_g"),
        ("flipper_length_mm", "body_mass_g"),
    ]

    num_plots = len(pairs)

    plt.figure(figsize=(12, 12))

    for i, (x, y) in enumerate(pairs, start=1):
        plt.subplot(2, 2, i)
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue="species",
            palette="deep",
            s=80,
            alpha=0.8
        )
        plt.title(f"{x} vs {y}")
        plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
