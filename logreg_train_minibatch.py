import sys
from describe import calculate_statistics, extract_numerical_columns
from utils import (
    read_csv_file,
    parse_csv_data,
    predict,
    normalization,
    save_model,
    shuffle_data,
    compute_log_loss,
    build_dataset
)

# =====================
# Training
# =====================

def mini_batch_gradient_step(batch_features, batch_labels, weights, bias, learning_rate):
    """
    Goal: Minimize the loss function by adjusting weights

    Algorithm (Mini-Batch GD):
    1. Start with random weights and bias
    2. For each mini-batch:
    - Compute predictions -> sigmoid(logit)
    - Compute error -> predictions - actual
    - Accumulate gradients over the batch
    - Average gradients and update weights
    """
    batch_size = len(batch_features)
    num_features = len(weights)
    weight_derivative = [0.0] * num_features
    bias_derivative = 0.0

    for student in range(batch_size):
        prediction = predict(batch_features[student], weights, bias)
        error = prediction - batch_labels[student]
        for feature in range(num_features):
            weight_derivative[feature] += error * batch_features[student][feature]
        bias_derivative += error

    weight_derivative = [grad / batch_size for grad in weight_derivative]
    bias_derivative /= batch_size

    for feature in range(num_features):
        weights[feature] -= learning_rate * weight_derivative[feature]
    bias -= learning_rate * bias_derivative
    return weights, bias


def train_mini_batch_gd(features_matrix, labels, learning_rate, epochs, batch_size=32):
    """
    Train a binary logistic regression model using mini-batch gradient descent.
    Updates weights after each mini-batch of samples.
    Returns trained weights and bias.
    """
    num_features = len(features_matrix[0])
    num_students = len(features_matrix)
    weights = [0.0] * num_features
    bias = 0.0

    for epoch in range(epochs):
        shuffled_features, shuffled_labels = shuffle_data(features_matrix, labels)
        for start in range(0, num_students, batch_size):
            end = min(start + batch_size, num_students)
            batch_features = shuffled_features[start:end]
            batch_labels = shuffled_labels[start:end]
            weights, bias = mini_batch_gradient_step(batch_features, batch_labels, weights, bias, learning_rate)

        if epoch % 40 == 0 or epoch == epochs - 1:
            loss = compute_log_loss(features_matrix, labels, weights, bias)
            print(f"     Epoch {epoch:>4}/{epochs} - Log Loss: {loss:.6f}")

    return weights, bias


def train_one_vs_rest(X, houses, house_names, learning_rate=0.05, epochs=200, batch_size=32):
    """
    Train one-vs-rest 4 separate binary classifiers using Mini-Batch GD:
   - Model 1: Gryffindor vs. Others
   - Model 2: Slytherin vs. Others
   - Model 3: Hufflepuff vs. Others
   - Model 4: Ravenclaw vs. Others
    """
    models = {}
    print(f"🏰 Houses: {house_names}\n")
    for house in house_names:
        y_binary = [1 if h == house else 0 for h in houses]
        n_positive = sum(y_binary)
        n_negative = len(y_binary) - n_positive
        print("                           ")
        print(f"   {house}: {n_positive} students")
        print(f"   Others: {n_negative} students")
        print("---------------------------")
        w, b = train_mini_batch_gd(X, y_binary, learning_rate, epochs, batch_size)
        models[house] = (w, b)
    return models


# =====================
# Main function
# =====================
def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train_minibatch.py <dataset_train.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    lines = read_csv_file(dataset_path)
    headers, rows = parse_csv_data(lines)
    label_col = "Hogwarts House"
    numerical_columns = extract_numerical_columns(headers, rows)
    numeric_feature_names = list(numerical_columns.keys())

    X, y = build_dataset(headers, rows, numeric_feature_names, label_col)
    print(f"✅ Dataset built successfully!")
    print(f"   Shape of X: ({len(X)}, {len(X[0]) if X else 0}) (students x features)")
    print(f"   Shape of y: ({len(y)},) (class labels, one per student)")
    print(f"   Students removed due to missing values: {len(rows) - len(X)}")

    stats = calculate_statistics(numerical_columns)
    means = [stats[f]["mean"] for f in numeric_feature_names]
    stds = [stats[f]["std"] for f in numeric_feature_names]
    X_scaled = normalization(X, means, stds)
    print("✅ Data normalized!")

    house_names = sorted(set(y))
    raw_models = train_one_vs_rest(
        X_scaled,
        y,
        house_names,
        learning_rate=0.05,
        epochs=200,
        batch_size=32
    )

    models = {}
    for house, params in raw_models.items():
        w, b = params
        models[house] = {"weights": w, "bias": b}
    save_model("model.json", models, means, stds, numeric_feature_names)
    print("Mini-Batch GD Training complete. Model saved to model.json")


if __name__ == "__main__":
    main()
