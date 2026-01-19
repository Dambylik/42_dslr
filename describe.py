import sys
import math


def read_csv_file(file_path):
    """Read CSV file and return lines."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    
    if len(lines) == 0:
        print("The dataset is empty.")
        sys.exit(1)
    
    return lines


def parse_csv_data(lines):
    """Parse CSV lines into headers and rows."""
    header_line = lines[0].strip()
    data_lines = lines[1:]
    headers = header_line.split(",")
    
    rows = []
    for line in data_lines:
        line = line.strip()
        if line == "":
            continue
        row = line.split(",")
        rows.append(row)
    
    return headers, rows


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
        if std == 0.0:
            std == 1.0
        min_value = calculate_min(sorted_values)
        max_value = calculate_max(sorted_values)    
        q25 = calculate_percentile(sorted_values, 0.25)
        q50 = calculate_percentile(sorted_values, 0.50)
        q75 = calculate_percentile(sorted_values, 0.75)
        
        stats[col_name] = {
            "count": count,
            "mean": mean,
            "std": std,
            "min": min_value,
            "25%": q25,
            "50%": q50,
            "75%": q75,
            "max": max_value
        }
    return stats


def print_statistics(stats):
    """Print statistics for all columns in tabular format."""
    if not stats:
        return

    columns = list(stats.keys())

    stat_labels = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    stat_keys   = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    stat_name_width = 12
    col_width = 15

    # Header row
    print(" " * stat_name_width, end="")
    for col in columns:
        print(col[:col_width].ljust(col_width), end="")
    print()

    # Data rows
    for label, key in zip(stat_labels, stat_keys):
        print(f"{label:<{stat_name_width}}", end="")
        for col in columns:
            value = stats[col][key]
            if isinstance(value, int):
                print(f"{value:<{col_width}}", end="")
            else:
                print(f"{value:<{col_width}.6f}", end="")
        print()


def main():
    """Main function of describe.py"""
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_name>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    lines = read_csv_file(file_path)
    headers, rows = parse_csv_data(lines)
    numerical_columns = extract_numerical_columns(headers, rows)
    stats = calculate_statistics(numerical_columns)
    print_statistics(stats)


if __name__ == "__main__":
    main()