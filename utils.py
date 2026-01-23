import pandas as pd
import sys
import math


def sigmoid(z):
    """Numerically stable sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ)"""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)


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

def load_data(path):
    print(f"path: {path}")
    try:
        df = pd.read_csv(path, index_col = 0)
    except:
        print("Fie error detected")
        sys.exit(1)
    print("df shape :", df.shape)
    features = df.columns.tolist()
    print("features :", features)
    return df, features
