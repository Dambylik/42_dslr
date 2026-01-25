import sys
import math
from utils import read_csv_file, parse_csv_data


def extract_numerical_columns(headers, rows):
    """Extract numerical data from rows, organized by column name."""
    numerical_columns = {}
    num_columns = len(headers)
    
    # Non-numerical columns to skip
    skip_columns = ["Index", "Hogwarts House", "First Name", "Last Name", "Birthday"]
    
    for col_index in range(num_columns):
        col_name = headers[col_index]
        if col_name in skip_columns:
            continue
        
        values = []
        for row in rows:
            if col_index >= len(row):
                continue
            cell = row[col_index].strip()
            if cell == "":
                continue
            try:
                value = float(cell)
                values.append(value)
            except ValueError:
                continue
        
        if len(values) > 0:
            numerical_columns[col_name] = values
    
    return numerical_columns


def calculate_count(values):
    """Calculate count of values."""
    return len(values)


def calculate_mean(values):
    """Calculate mean of values."""
    total = 0.0
    for v in values:
        total += v
    return total / len(values)


def calculate_min(values):
    """Calculate minimum value."""
    min_value = values[0]
    return min_value


def calculate_max(values):
    """Calculate maximum value."""
    max_value = values[-1]
    return max_value


def calculate_std(values, mean):
    """Calculate standard deviation."""
    squared_diff_sum = 0.0
    for v in values:
        diff = v - mean
        squared_diff_sum += diff * diff
    variance = squared_diff_sum / len(values)
    return math.sqrt(variance)


def calculate_percentile(sorted_values, p):
    """Calculate percentile from sorted values."""
    n = len(sorted_values)
    if n == 0:
        return None

    position = p * (n - 1)
    lower_index = int(position)
    upper_index = lower_index + 1

    if upper_index >= n:
        return sorted_values[lower_index]

    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + fraction * (upper_value - lower_value)


# =====================
# Bonus statistics
# =====================
def calculate_variance(values, mean):
    """Calculate variance (std squared)."""
    squared_diff_sum = 0.0
    for v in values:
        diff = v - mean
        squared_diff_sum += diff * diff
    return squared_diff_sum / len(values)


def calculate_range(min_val, max_val):
    """Calculate range (max - min)."""
    return max_val - min_val


def calculate_iqr(q25, q75):
    """Calculate interquartile range (75% - 25%)."""
    return q75 - q25


def calculate_skewness(values, mean, std):
    """
    Calculate skewness (measure of asymmetry).
    Skewness = E[(X - μ)³] / σ³
    """
    if std == 0:
        return 0.0
    n = len(values)
    cubed_diff_sum = 0.0
    for v in values:
        diff = (v - mean) / std
        cubed_diff_sum += diff * diff * diff
    return cubed_diff_sum / n


def calculate_kurtosis(values, mean, std):
    """
    Calculate kurtosis (measure of tailedness).
    Kurtosis = E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
    """
    if std == 0:
        return 0.0
    n = len(values)
    fourth_diff_sum = 0.0
    for v in values:
        diff = (v - mean) / std
        fourth_diff_sum += diff * diff * diff * diff
    return (fourth_diff_sum / n) - 3


def calculate_statistics(numerical_columns):
    """Calculate statistics for all numerical columns."""
    stats = {}

    for col_name, values in numerical_columns.items():
        if not values:
            continue # skip empty columns

        sorted_values = sorted(values)
        count = calculate_count(sorted(sorted_values))
        mean = calculate_mean(sorted_values)
        std = calculate_std(sorted_values, mean)
        std_for_calc = std if std != 0.0 else 1.0
        min_value = calculate_min(sorted_values)
        max_value = calculate_max(sorted_values)
        q25 = calculate_percentile(sorted_values, 0.25)
        q50 = calculate_percentile(sorted_values, 0.50)
        q75 = calculate_percentile(sorted_values, 0.75)

        # Bonus statistics
        variance = calculate_variance(sorted_values, mean)
        range_val = calculate_range(min_value, max_value)
        iqr = calculate_iqr(q25, q75)
        skewness = calculate_skewness(sorted_values, mean, std_for_calc)
        kurtosis = calculate_kurtosis(sorted_values, mean, std_for_calc)

        if std == 0.0:
            std = 1.0

        stats[col_name] = {
            "count": count,
            "mean": mean,
            "std": std,
            "var": variance,
            "min": min_value,
            "25%": q25,
            "50%": q50,
            "75%": q75,
            "max": max_value,
            "range": range_val,
            "iqr": iqr,
            "skew": skewness,
            "kurt": kurtosis
        }
    return stats


def print_statistics(stats, output_file=None):
    """Print statistics for all columns in tabular format."""
    if not stats:
        return

    columns = list(stats.keys())

    # Original + Bonus fields
    stat_labels = ["Count", "Mean", "Std", "Var", "Min", "25%", "50%", "75%", "Max", "Range", "IQR", "Skew", "Kurt"]
    stat_keys   = ["count", "mean", "std", "var", "min", "25%", "50%", "75%", "max", "range", "iqr", "skew", "kurt"]

    stat_name_width = 12
    col_width = 18

    lines = []

    header = " " * stat_name_width
    for col in columns:
        header += col[:col_width].ljust(col_width)
    lines.append(header)

    for label, key in zip(stat_labels, stat_keys):
        row = f"{label:<{stat_name_width}}"
        for col in columns:
            value = stats[col][key]
            if isinstance(value, int):
                row += f"{value:<{col_width}}"
            else:
                row += f"{value:<{col_width}.6f}"
        lines.append(row)

    for line in lines:
        print(line)

    if output_file:
        with open(output_file, "w") as f:
            f.write("DESCRIBE OUTPUT\n")
            f.write("=" * 80 + "\n\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")

            # Vertical format for easier reading
            f.write("\n" + "=" * 80 + "\n")
            f.write("VERTICAL FORMAT \n")
            f.write("=" * 80 + "\n\n")
            for col in columns:
                f.write(f"\n{col}\n")
                f.write("-" * 40 + "\n")
                for label, key in zip(stat_labels, stat_keys):
                    value = stats[col][key]
                    if isinstance(value, int):
                        f.write(f"  {label:<10} {value}\n")
                    else:
                        f.write(f"  {label:<10} {value:.6f}\n")
        print(f"\nOutput saved to: {output_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_name>")
        sys.exit(1)

    file_path = sys.argv[1]
    lines = read_csv_file(file_path)
    headers, rows = parse_csv_data(lines)
    numerical_columns = extract_numerical_columns(headers, rows)
    stats = calculate_statistics(numerical_columns)
    print_statistics(stats, output_file="describe_output.txt")


if __name__ == "__main__":
    main()