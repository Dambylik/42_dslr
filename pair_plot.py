import matplotlib.pyplot as plt
import pandas as pd
import sys


def pair(data):
    hue_col = "Hogwarts House"
    features = data.select_dtypes(include="number").columns
    houses = data[hue_col].dropna().unique()

    n = len(features)
    fig_size = (1.5 * 1.5 * n, 1.5 * n)
    fig, axes = plt.subplots(n, n, figsize=fig_size)

    for row, y_var in enumerate(features):
        for col, x_var in enumerate(features):
            ax = axes[row, col]

            if row == col:
                for house in houses:
                    values = data[data[hue_col] == house][x_var].dropna()
                    ax.hist(values, bins=20, alpha=0.6)
            else:
                for house in houses:
                    subset = data[data[hue_col] == house]
                    ax.scatter(
                        subset[x_var],
                        subset[y_var],
                        s=15,
                        alpha=0.6
                    )

            if row == n - 1:
                ax.set_xlabel(x_var, rotation=45, ha="right")
            else:
                ax.set_xticks([])

            if col == 0:
                ax.set_ylabel(y_var, rotation=45, ha="right")
            else:
                ax.set_yticks([])

    fig.subplots_adjust(left=0.08, bottom=0.12, wspace=0.05, hspace=0.05)
    plt.show()
    print("done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py [dataset_filename]")
    else:
        data = pd.read_csv(sys.argv[1]).set_index('Index')
        numeric_features = data.select_dtypes(include='number').columns
        numeric_features = numeric_features.append(
            pd.Index([data['Hogwarts House']]))
        pair(data)
