import sys
from describe import calculate_statistics, extract_numerical_columns
from utils import (
    read_csv_file,
    parse_csv_data,
    predict,
    normalization,
    save_model,
    compute_log_loss,
    build_dataset
)

# =====================
# Training
# =====================

def gradient_descent_step(features_matrix, labels, weights, bias, learning_rate):
    """
    Goal: Minimize the loss function by adjusting weights

    Algorithm:
    1. Start with random weights and bias
    2. For each iteration:
    - Calculate linear score "logit" -> single number
    - Compute predictions -> sigmoid(logit)
    - Compute error -> predictions - actual
    - Compute gradients
    - Update weights and bias
    """
    num_students = len(features_matrix)
    num_features = len(weights)
    weight_derivative = [0.0] * num_features
    bias_derivative = 0.0

    for student in range(num_students):
        prediction = predict(features_matrix[student], weights, bias)
        error = prediction - labels[student]
        for feature in range(num_features):
            weight_derivative[feature] += error * features_matrix[student][feature]
        bias_derivative += error

    weight_derivative = [grad / num_students for grad in weight_derivative]
    bias_derivative /= num_students

    for feature in range(num_features):
        weights[feature] -= learning_rate * weight_derivative[feature]
    bias -= learning_rate * bias_derivative
    return weights, bias


def train_logistic_regression(features_matrix, labels, learning_rate, epochs):
    """
    Train a binary logistic regression model using batch gradient descent.
    Returns trained weights and bias.
    """
    num_features = len(features_matrix[0])
    weights = [0.0] * num_features
    bias = 0.0
    for epoch in range(epochs):
        weights, bias = gradient_descent_step(features_matrix, labels, weights, bias, learning_rate)
        if epoch % 200 == 0 or epoch == epochs - 1:
            loss = compute_log_loss(features_matrix, labels, weights, bias)
            print(f"     Epoch {epoch:>4}/{epochs} - Log Loss: {loss:.6f}")
    return weights, bias


def train_one_vs_rest(X, houses, house_names, learning_rate = 0.1, epochs = 1000):
    """
    Train one-vs-rest 4 separate binary classifiers:
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
        w, b = train_logistic_regression(X, y_binary, learning_rate, epochs)
        models[house] = (w, b)
    return models


# =====================
# Main function
# =====================
def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>")
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
        learning_rate=0.01,
        epochs=500
    )

    models = {}
    for house, params in raw_models.items():
        w, b = params
        models[house] = {"weights": w, "bias": b}
    save_model("model.json", models, means, stds, numeric_feature_names)
    print("Training complete. Model saved to model.json")


if __name__ == "__main__":
    main()
