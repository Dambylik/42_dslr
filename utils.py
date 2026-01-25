import pandas as pd
import sys
import math
import json
import random


# =====================
# Math utilities
# =====================
def sigmoid(z):
    """Numerically stable sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ)"""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)


def log_loss(y, y_hat, eps=1e-15):
    """
    Binary cross-entropy loss.
    y: true label (0 or 1)
    y_hat: predicted probability
    eps: small value to avoid log(0)
    """
    y_hat = max(eps, min(1 - eps, y_hat))
    return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))


# =====================
# CSV utilities
# =====================
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
    """Load data using pandas (legacy function)."""
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


# =====================
# Normalization utilities
# =====================
def normalization(X, means, stds):
    """
    Normalize dataset using z-score normalization: x_scaled = (x - μ) / σ
    X: list of feature vectors (2D list)
    means: list of mean values for each feature
    stds: list of standard deviation values for each feature
    Returns: normalized dataset (2D list)
    """
    X_scaled = []
    for row in X:
        scaled_row = []
        for j in range(len(row)):
            scaled_value = (row[j] - means[j]) / stds[j]
            scaled_row.append(scaled_value)
        X_scaled.append(scaled_row)
    return X_scaled


def normalize(x, means, stds):
    """
    Normalize a single feature vector using z-score normalization.
    x: single feature vector (list)
    means: list of mean values for each feature
    stds: list of standard deviation values for each feature
    Returns: normalized feature vector (list)
    """
    return [(x[i] - means[i]) / stds[i] for i in range(len(x))]


# =====================
# Data shuffling
# =====================
def shuffle_data(X, y):
    """
    Shuffle X and y together using Fisher-Yates algorithm.
    X: list of feature vectors
    y: list of labels
    Returns: shuffled X and y (as tuple)
    """
    n = len(X)
    X_shuffled = X[:]
    y_shuffled = y[:]
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        X_shuffled[i], X_shuffled[j] = X_shuffled[j], X_shuffled[i]
        y_shuffled[i], y_shuffled[j] = y_shuffled[j], y_shuffled[i]
    return X_shuffled, y_shuffled


# =====================
# Model persistence
# =====================
def save_model(path, models, means, stds, feature_names):
    """
    Save model to JSON file.
    path: output file path
    models: dict of house -> {"weights": [...], "bias": float}
    means: list of mean values for normalization
    stds: list of std values for normalization
    feature_names: list of feature names
    """
    payload = {
        "models": models,
        "means": means,
        "stds": stds,
        "features": feature_names
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_model(path):
    """
    Load model from JSON file.
    path: model file path
    Returns: tuple of (models, means, stds, feature_names)
    """
    with open(path, "r") as f:
        model = json.load(f)
    return model["models"], model["means"], model["stds"], model["features"]
