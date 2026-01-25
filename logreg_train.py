import sys
from describe import calculate_statistics, extract_numerical_columns
from utils import (
    read_csv_file,
    parse_csv_data,
    sigmoid,
    normalization,
    save_model
)


def gradient_descent_step(X, y, w, b, learning_rate):
    """
    Perform one batch gradient descent step.
    X: list of feature vectors, 2 students and 2 features
    X = [
        [2.0, 1.0],   # student A
        [1.0, 3.0]    # student B
        ]
    y: list of true labels (0 or 1), if we're training "House = Gryffindor".
        y = [1, # student A is Gryffindor
             0]  # student B is not

    w: list of weights start neutral
    b: bias (float)
        w = [0.0, 0.0]
        b = 0.0
        learning_rate = 0.1
    d: derivative
    """
    m = len(X)          # 2 students
    n = len(w)          # 2 features
    dw = [0.0] * n      # gradient of the loss with respect to the weights
    db = 0.0            # gradient of the loss with respect to the bias

    for i in range(m):
        # linear score
        z = sum(w[j] * X[i][j] for j in range(n)) + b
        # prediction
        y_hat = sigmoid(z)
        # error
        error = y_hat - y[i]
        # accumulate gradients
        for j in range(n):
            dw[j] += error * X[i][j]
        db += error
    # average gradients
    dw = [g / m for g in dw]
    db /= m
    # update parameters
    for j in range(n):
        w[j] -= learning_rate * dw[j]
    b -= learning_rate * db
    return w, b

# =====================
# Training
# =====================

def train_logistic_regression(X, y, learning_rate, epochs):
    """
    Train a binary logistic regression model using batch gradient descent.
    Returns trained weights and bias.
    """
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    # Training loop
    for _ in range(epochs):
        w, b = gradient_descent_step(X, y, w, b, learning_rate)
    return w, b


def train_one_vs_rest(X, houses, house_names, learning_rate = 0.1, epochs = 1000):
    """
    Train one-vs-rest logistic regression for a specific house.
    houses: list of house labels for each student
    house_name: the house we are training for (e.g., "Gryffindor")
    """
    models = {}
    for house in house_names:
        # Build binary labels
        y_binary = [1 if h == house else 0 for h in houses]
        w, b = train_logistic_regression(X, y_binary, learning_rate, epochs)
        models[house] = (w, b)
    return models


# =====================
# Main pipeline
# =====================
def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Load data
    lines = read_csv_file(dataset_path)
    headers, rows = parse_csv_data(lines)

    label_col = "Hogwarts House"

    # Extract numeric columns
    numerical_columns = extract_numerical_columns(headers, rows)
    numeric_feature_names = list(numerical_columns.keys())
    # Build X (list of feature vectors) and y (labels)
    X = []
    y = []
    for row in rows:
        try:
            features = [float(row[headers.index(f)]) if row[headers.index(f)].strip() != '' else float('nan') for f in numeric_feature_names]
            if any([str(x) == 'nan' for x in features]):
                continue  # skip rows with missing values in numeric features
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

    # Train one-vs-rest
    house_names = sorted(set(y))
    raw_models = train_one_vs_rest(
        X_scaled,
        y,
        house_names,
        learning_rate=0.01,
        epochs=1000
    )

    # Convert each model to dict with 'weights' and 'bias' keys
    models = {}
    for house, params in raw_models.items():
        w, b = params
        models[house] = {"weights": w, "bias": b}

    # Save model with only numeric features
    save_model("model.json", models, means, stds, numeric_feature_names)
    print("Training complete. Model saved to model.json")


if __name__ == "__main__":
    main()
