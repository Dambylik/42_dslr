import sys
import matplotlib.pyplot as plt
from utils import read_csv_file, parse_csv_data
from describe import calculate_mean, calculate_std, extract_numerical_columns_NAN
import pandas as pd
import math

def ft_covariance(x, y, x_mean, y_mean):
    value = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    return value / (len(x) - 1)

def pearson_equation(x, y):
    """
    Calculate Pearson correlation
    """
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if len(pairs) < 2:
        return 0.0
    x_clean = [p[0] for p in pairs]
    y_clean = [p[1] for p in pairs]
    x_mean = calculate_mean(x_clean)
    y_mean = calculate_mean(y_clean)
    cov = ft_covariance(x_clean, y_clean, x_mean, y_mean)
    if (calculate_std(x_clean, x_mean)) == 0 or (calculate_std(y_clean, y_mean)) == 0:
        return 0
    return cov / (calculate_std(x_clean, x_mean) * calculate_std(y_clean, y_mean))

def calculate_corr(numerical_columns):
    """
    numerical_columns: dictionary from extract_numerical_columns
    """
    matrix = {}
    headers = sorted(numerical_columns.keys())

    for col1 in headers:
        matrix[col1] = {}
        for col2 in headers:
            # We pass the lists of numbers directly from the dict
            corr_value = pearson_equation(numerical_columns[col1], numerical_columns[col2])
            matrix[col1][col2] = corr_value
    return matrix

def extract_pair_by_house(rows, headers, feature_x, feature_y):
    """
    Extract pairs by houses
    """
    house_index = headers.index("Hogwarts House")
    x_index = headers.index(feature_x)
    y_index = headers.index(feature_y)

    houses = {
        "Gryffindor": {"x": [], "y": []},
        "Ravenclaw": {"x": [], "y": []},
        "Hufflepuff": {"x": [], "y": []},
        "Slytherin": {"x": [], "y": []}
    }

    for row in rows:
        if len(row) <= max(house_index, x_index, y_index):
            continue

        house = row[house_index].strip()
        x_val = row[x_index].strip()
        y_val = row[y_index].strip()

        if house not in houses or x_val == "" or y_val == "":
            continue

        try:
            houses[house]["x"].append(float(x_val))
            houses[house]["y"].append(float(y_val))
        except ValueError:
            continue

    return houses


def plot_scatter(houses_data, feature_x, feature_y):
    plt.figure(figsize=(10, 6))

    for house, values in houses_data.items():
        if not values["x"]:
            continue
        plt.scatter(
            values["x"],
            values["y"],
            alpha=0.6,
            s=15,
            label=house
        )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{feature_x} vs {feature_y}")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    lines = read_csv_file(file_path)
    headers, rows = parse_csv_data(lines)
    numerical_columns = extract_numerical_columns_NAN(headers, rows)
    print("-" * 50)
    matrix = calculate_corr(numerical_columns)
    df = pd.DataFrame(matrix).reindex(sorted(matrix.keys()))
    print(df.round(4))
    print("-" * 50)
    #test CORRELATION
    # df = pd.read_csv(file_path)
    # if 'Index' in df.columns:
    #     df = df.drop(columns=['Index'])
    # numerical_df = df.select_dtypes(include=['number'])
    # corr_matrix = numerical_df.corr()
    # corr_matrix = corr_matrix.sort_index(axis=0).sort_index(axis=1)
    # print(corr_matrix.round(4))
    feature_x = "Astronomy"
    feature_y = "Defense Against the Dark Arts"  # Correlation r=-1.0 (most similar)

    houses_data = extract_pair_by_house(rows, headers, feature_x, feature_y)
    plot_scatter(houses_data, feature_x, feature_y)


if __name__ == "__main__":
    main()

