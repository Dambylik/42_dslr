import sys
from describe import calculate_statistics, extract_numerical_columns
from utils import (
    read_csv_file,
    parse_csv_data,
    sigmoid,
    normalization,
    save_model,
    shuffle_data
)


def sgd_step(x, y_true, w, b, learning_rate):
    """
    Perform one stochastic gradient descent step (single sample update).
    x: single feature vector (list of floats)
    y_true: single label (0 or 1)
    w: list of weights
    b: bias (float)
    learning_rate: step size
    """
    n = len(w)
    # linear score
    z = sum(w[j] * x[j] for j in range(n)) + b
    # prediction
    y_hat = sigmoid(z)
    # error
    error = y_hat - y_true
    # update parameters immediately (no accumulation)
    for j in range(n):
        w[j] -= learning_rate * error * x[j]
    b -= learning_rate * error
    return w, b


# =====================
# Training
# =====================
def train_sgd(X, y, learning_rate, epochs):
    """
    Train a binary logistic regression model using stochastic gradient descent.
    Updates weights after each sample, shuffles data each epoch.
    Returns trained weights and bias.
    """
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    # Training loop
    for _ in range(epochs):
        # Shuffle data each epoch for better convergence
        X_shuffled, y_shuffled = shuffle_data(X, y)
        # Update after each sample
        for i in range(len(X_shuffled)):
            w, b = sgd_step(X_shuffled[i], y_shuffled[i], w, b, learning_rate)
    return w, b


def train_one_vs_rest(X, houses, house_names, learning_rate=0.1, epochs=100):
    """
    Train one-vs-rest logistic regression using SGD.
    houses: list of house labels for each student
    house_names: list of unique house names
    """
    models = {}
    for house in house_names:
        # Build binary labels
        y_binary = [1 if h == house else 0 for h in houses]
        w, b = train_sgd(X, y_binary, learning_rate, epochs)
        models[house] = (w, b)
    return models


# =====================
# Main pipeline
# =====================
def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train_sgd.py <dataset_train.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Load data
    lines = read_csv_file(dataset_path)
    headers, rows = parse_csv_data(lines)

    label_col = "Hogwarts House"

    # Extract numeric columns
    numerical_columns = extract_numerical_columns(headers, rows)
    numeric_feature_names = list(numerical_columns.keys())

    # Build X (feature vectors) and y (labels)
    X = []
    y = []
    for row in rows:
        try:
            features = [float(row[headers.index(f)]) if row[headers.index(f)].strip() != '' else float('nan') for f in numeric_feature_names]
            if any([str(x) == 'nan' for x in features]):
                continue
            label = row[headers.index(label_col)]
        except Exception:
            continue
        X.append(features)
        y.append(label)

    # Compute statistics
    stats = calculate_statistics(numerical_columns)
    means = [stats[f]["mean"] for f in numeric_feature_names]
    stds = [stats[f]["std"] for f in numeric_feature_names]

    # Normalize
    X_scaled = normalization(X, means, stds)

    # Train one-vs-rest using SGD
    house_names = sorted(set(y))
    raw_models = train_one_vs_rest(
        X_scaled,
        y,
        house_names,
        learning_rate=0.1,
        epochs=100
    )

    # Convert to dict format
    models = {}
    for house, params in raw_models.items():
        w, b = params
        models[house] = {"weights": w, "bias": b}

    # Save model
    save_model("model.json", models, means, stds, numeric_feature_names)
    print("SGD Training complete. Model saved to model.json")


if __name__ == "__main__":
    main()
