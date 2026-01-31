import sys
import math
from utils import read_csv_file, parse_csv_data


def extract_numerical_columns(headers, rows):
    """Extract numerical data from rows, organized by column name."""
    numerical_columns = {}
    num_columns = len(headers)
    
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
            continue

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


def print_statistics(stats, output_file=None, file_path=None, headers=None, rows=None):
    """Print statistics for all columns in tabular format."""
    if not stats:
        return

    columns = list(stats.keys())
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

    if output_file:
        with open(output_file, "w") as f:
            if file_path and headers and rows:
                # Calculate total students and features
                total_students = len(rows)
                total_features = len(headers)

                f.write(f"📚 Total students: {total_students}\n")
                f.write(f"📊 Number of features: {total_features}\n")

                # Calculate house distribution
                house_col_index = headers.index("Hogwarts House") if "Hogwarts House" in headers else None
                if house_col_index is not None:
                    house_counts = {}
                    for row in rows:
                        if house_col_index < len(row):
                            house = row[house_col_index].strip()
                            if house:
                                house_counts[house] = house_counts.get(house, 0) + 1

                    f.write(f"\n🏠 Number of students. Distribution by Houses:\n")
                    for house in sorted(house_counts.keys()):
                        f.write(f"{house}    {house_counts[house]}\n")

                # Calculate missing values for numerical features
                numerical_features = list(stats.keys())
                missing_data = []

                for feature in numerical_features:
                    if feature not in headers:
                        continue

                    feature_index = headers.index(feature)
                    missing_count = 0

                    for row in rows:
                        if feature_index >= len(row):
                            missing_count += 1
                        elif row[feature_index].strip() == "":
                            missing_count += 1
                        else:
                            try:
                                float(row[feature_index])
                            except ValueError:
                                missing_count += 1

                    if missing_count > 0:
                        missing_pct = round(missing_count / total_students * 100, 2)
                        missing_data.append((feature, missing_count, missing_pct))

                # Sort by missing count descending
                missing_data.sort(key=lambda x: x[1], reverse=True)

                f.write("\n")
                if len(missing_data) > 0:
                    total_missing = sum(count for _, count, _ in missing_data)
                    f.write(f"⚠️  Features with missing values : {total_missing}\n")
                    f.write(f"{'Feature':<30} {'Missing Count':<15} {'Missing %':<10}\n")
                    for feature, count, pct in missing_data:
                        f.write(f"{feature:<30} {count:<15} {pct:<10}\n")
                else:
                    f.write("✅ No missing values found!\n")

                f.write("\n" + "=" * 80 + "\n\n")

            f.write("OUTPUT\n")
            f.write("=" * 80 + "\n\n")
            for line in lines:
                f.write(line + "\n")
            f.write("\n")
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
    print_statistics(stats, output_file="describe_output.txt", file_path=file_path, headers=headers, rows=rows)


if __name__ == "__main__":
    main()