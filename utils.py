import pandas as pd
import sys
import math
import json
import random


# =====================
# Math utilities
# =====================
def sigmoid(z):
    """Sigmoid function"""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)


def predict(features, weights, bias):
    """Compute prediction.
    Hypothesis = sigmoid function of linear combination of features and weights"""
    logit = sum(weights[j] * features[j] for j in range(len(weights))) + bias
    return sigmoid(logit)


def compute_log_loss(features_matrix, labels, weights, bias):
    """
    Compute the log loss (binary cross-entropy) over all students.
    Cost function: 
    J(θ) =  -1/num_students x Σ [real_answer x log(prediction) + (1-real_answer) x log(1-prediction)]
    Lower = better. Perfect model -> 0.0
    """
    num_students = len(features_matrix)
    total_loss = 0.0
    for student in range(num_students):
        prediction = predict(features_matrix[student], weights, bias)
        prediction = max(min(prediction, 1 - 1e-15), 1e-15)
        total_loss += labels[student] * math.log(prediction) + (1 - labels[student]) * math.log(1 - prediction)
    return -total_loss / num_students


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
    """Load data from CSV file into DataFrame and extract features."""
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
    Normalize using z-score normalization.
    
    Args:
        X: single feature vector (list) or list of feature vectors (2D list)
        means: list of mean values for each feature
        stds: list of standard deviation values for each feature

    Returns:
        normalized vector (list) or normalized dataset (2D list)
    """
    if len(X) == 0:
        return X
    if isinstance(X[0], list):
        return [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in X]
    return [(X[i] - means[i]) / stds[i] for i in range(len(X))]


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
# Dataset building
# =====================
def build_dataset(headers, rows, feature_names, label_col='Hogwarts House'):
    """
    Build X (feature matrix) and y (labels) from raw CSV rows.
    Skips rows with missing values in numeric features.
    Returns X (list of feature vectors), y (list of labels).
    """
    X = []
    y = []
    for row in rows:
        try:
            features = [
                float(row[headers.index(f)]) if row[headers.index(f)].strip() != '' else float('nan')
                for f in feature_names
            ]
            if any(str(x) == 'nan' for x in features):
                continue
            label = row[headers.index(label_col)]
        except Exception:
            continue
        X.append(features)
        y.append(label)
    return X, y


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
        json.dump(payload, f, indent=4)


def load_model(path):
    """
    Load model from JSON file.
    path: model file path
    Returns: tuple of (models, means, stds, feature_names)
    """
    with open(path, "r") as f:
        model = json.load(f)
    return model["models"], model["means"], model["stds"], model["features"]
