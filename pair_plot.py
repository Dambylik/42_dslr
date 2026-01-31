import sys
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data


def plot_pair(df):
    sns.pairplot(
        df,
        hue="Hogwarts House",
        diag_kind="hist",
        plot_kws={"alpha": 0.6, "s": 15}
    )
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset.csv")
        sys.exit(1)
    path = sys.argv[1]
    df, _ = load_data(path)
    drop_cols = ["First Name", "Last Name", "Birthday"]
    df = df.drop(columns=drop_cols, errors='ignore')
    plot_pair(df)


if __name__ == "__main__":
    main()