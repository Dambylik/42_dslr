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


def mini_batch_gradient_step(X_batch, y_batch, w, b, learning_rate):
    """
    Perform one mini-batch gradient descent step.
    Computes gradients over a small batch of samples, then updates weights.

    X_batch: list of feature vectors (mini-batch)
    y_batch: list of true labels (0 or 1)
    w: list of weights
    b: bias (float)
    learning_rate: step size
    """
    m = len(X_batch)    # mini-batch size
    n = len(w)          # number of features
    dw = [0.0] * n      # gradient of the loss with respect to the weights
    db = 0.0            # gradient of the loss with respect to the bias

    # Accumulate gradients over mini-batch
    for i in range(m):
        # linear score
        z = sum(w[j] * X_batch[i][j] for j in range(n)) + b
        # prediction
        y_hat = sigmoid(z)
        # error
        error = y_hat - y_batch[i]
        # accumulate gradients
        for j in range(n):
            dw[j] += error * X_batch[i][j]
        db += error

    # Average gradients over mini-batch
    dw = [g / m for g in dw]
    db /= m

    # Update parameters
    for j in range(n):
        w[j] -= learning_rate * dw[j]
    b -= learning_rate * db
    return w, b


# =====================
# Training
# =====================
def train_mini_batch_gd(X, y, learning_rate, epochs, batch_size=32):
    """
    Train a binary logistic regression model using mini-batch gradient descent.
    Updates weights after each mini-batch of samples.
    Returns trained weights and bias.
    """
    n_features = len(X[0])
    n_samples = len(X)
    w = [0.0] * n_features
    b = 0.0

    # Training loop
    for _ in range(epochs):
        # Shuffle data each epoch
        X_shuffled, y_shuffled = shuffle_data(X, y)

        # Process mini-batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            w, b = mini_batch_gradient_step(X_batch, y_batch, w, b, learning_rate)

    return w, b


def train_one_vs_rest(X, houses, house_names, learning_rate=0.05, epochs=200, batch_size=32):
    """
    Train one-vs-rest logistic regression using Mini-Batch GD.
    houses: list of house labels for each student
    house_names: list of unique house names
    batch_size: number of samples per mini-batch
    """
    models = {}
    for house in house_names:
        # Build binary labels
        y_binary = [1 if h == house else 0 for h in houses]
        w, b = train_mini_batch_gd(X, y_binary, learning_rate, epochs, batch_size)
        models[house] = (w, b)
    return models


# =====================
# Main pipeline
# =====================
def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train_minibatch.py <dataset_train.csv>")
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

    # Train one-vs-rest using Mini-Batch GD
    house_names = sorted(set(y))
    raw_models = train_one_vs_rest(
        X_scaled,
        y,
        house_names,
        learning_rate=0.05,
        epochs=200,
        batch_size=32
    )

    # Convert to dict format
    models = {}
    for house, params in raw_models.items():
        w, b = params
        models[house] = {"weights": w, "bias": b}

    # Save model
    save_model("model.json", models, means, stds, numeric_feature_names)
    print("Mini-Batch GD Training complete. Model saved to model.json")


if __name__ == "__main__":
    main()
