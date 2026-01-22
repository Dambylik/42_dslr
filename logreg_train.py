import sys
import math
from describe import read_csv_file, parse_csv_data



def sigmoid(z):
    """sigmoid function : σ(z) = 1 / (1 + e⁻ᶻ)"""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)


def log_loss(y, y_hat, eps=1e-15):
    """
    y_hat is computed with
    1. Linear score: z = w · x + b
    2. Sigmoid: y_hat = sigmoid(z)
    """
    # clip predictions to avoid log(0)
    y_hat = max(eps, min(1 - eps, y_hat))
    return -(y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat))


def gradient_descent_step(X, y, w, b, learning_rate):
    """
    Perform one batch gradient descent step.
    X: list of feature vectors, 2 students and 2 features
    X = [
        [2.0, 1.0],   # student A
        [1.0, 3.0]    # student B
        ]
    y: list of true labels (0 or 1), if we’re training “House = Gryffindor”.
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


def predict_binary(x, w, b, threshold = 0.5):
    """Predict binary label for a single data point x"""
    z = sum(w[j] * x[j] for j in range(len(w))) + b
    y_hat = sigmoid(z)
    return 1 if y_hat >= threshold else 0


def compute_accuracy(X, y, w, b):
    """Compute accuracy on dataset X with true labels y"""
    correct_predictions = 0
    m = len(X)
    for i in range(m):
        y_pred = predict_binary(X[i], w, b)
        if y_pred == y[i]:
            correct_predictions += 1
    return correct_predictions / m


def train_logistic_regression(X, y, learning_rate, epochs):
    """
    Train a binary logistic regression model using batch gradient descent.
    Returns trained weights and bias.
    """
    m = len(X)
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
        #Build binary labels
        y_binary = [1 if h == house else 0 for h in houses]

        w, b = train_logistic_regression(X, y_binary, learning_rate, epochs)

        models[house] = (w,b)
    return models
   

def predict_house(x, models):
    """Predict house for one student usoing OVR model"""
    best_house = None
    best_prob = -1
    for house, (w, b) in models.items():
        z = sum(w[j] * x[j] for j in range(len(w))) + b
        prob = sigmoid(z)
        if prob > best_prob:
            best_prob = prob
            best_house = house
    return best_house   


def evaluate_one_vs_rest(X, true_house, models):
    """Evaluate OVR model accuracy on dataset"""
    correct_predictions = 0
    total = len(X)
    for i in range(total):
        predicted_house = predict_house(X[i], models)
        if predicted_house == true_house[i]:
            correct_predictions += 1
    accuracy = correct_predictions / total
    return accuracy


def evaluate_per_house(X, true_houses, models, house_names):
    """Evaluate accuracy per house in OVR model"""
    stats = {house: {
        "correct": 0,
        "total": 0
    } for house in house_names}

    for i in range(len(X)):
        true_house = true_houses[i]
        predicted = predict_house(X[i], models)
        stats[true_house]["total"] += 1
        if predicted == true_house:
            stats[true_house]["correct"] += 1

    for house in house_names:
        total = stats[house]["total"]
        correct = stats[house]["correct"]
        accuracy = correct / total if total > 0 else 0
        print(f"{house} accuracy: {accuracy:.2f}")


def normalization(X, means, stds):
    """ Normalize the dataset using z-score normalization x_scaled = (x - μ) / σ"""
    """mu (μ) = = mean of the feature"""
    """sigma (σ) = standard deviation of the feature
    X_scaled = [
    [(x₁₁ - μ₁)/σ₁, (x₁₂ - μ₂)/σ₂, ...],
    [(x₂₁ - μ₁)/σ₁, (x₂₂ - μ₂)/σ₂, ...],
    ]
    """
    X_scaled = []
    for row in X:
        scaled_row = []
        for j in range(len(row)):
            scaled_value = (row[j] - means[j]) / stds[j]
            scaled_row.append(scaled_value)
        X_scaled.append(scaled_row)
    return X_scaled
        

def main():
    # ---------- 1. Load dataset ----------
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    lines = read_csv_file(file_path)
    headers, rows = parse_csv_data(lines)

    # ---------- 2. Define features ----------
    label_col = "Hogwarts House"
    skip_cols = ["Index", "First Name", "Last Name", "Birthday", label_col, "Best Hand"]
    feature_names = [h for h in headers if h not in skip_cols]
    house_names = ["Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"]

    # ---------- 3. Extract features and labels ----------
    X = []
    y = []
    for row in rows:
        try:
            features = [float(row[headers.index(f)]) if row[headers.index(f)] != '' else 0.0 for f in feature_names]
            label = row[headers.index(label_col)]
            if label not in house_names:
                continue
            X.append(features)
            y.append(label)
        except ValueError:
            continue

    # ---------- 4. Train / test split ----------
    # Simple split: last 20% for test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ---------- 5. Compute statistics on TRAIN ----------
    from describe import calculate_statistics
    numerical_columns = {feature_names[i]: [row[i] for row in X_train] for i in range(len(feature_names))}
    stats = calculate_statistics(numerical_columns)
    means = [stats[f]["mean"] for f in feature_names]
    stds = [stats[f]["std"] if stats[f]["std"] != 0 else 1.0 for f in feature_names]

    # ---------- 6. Normalize ----------
    X_train_scaled = normalization(X_train, means, stds)
    X_test_scaled = normalization(X_test, means, stds)

    # ---------- 7. Train model ----------
    models = train_one_vs_rest(X_train_scaled, y_train, house_names, learning_rate=0.1, epochs=1000)

    # ---------- 8. Evaluate ----------
    print("Train accuracy:", evaluate_one_vs_rest(X_train_scaled, y_train, models))
    print("Test accuracy:", evaluate_one_vs_rest(X_test_scaled, y_test, models))
    evaluate_per_house(X_test_scaled, y_test, models, house_names)

    # ---------- 9. Save model ----------
    import pickle
    with open("logreg_model.pkl", "wb") as f:
        pickle.dump({
            "models": models,
            "feature_names": feature_names,
            "means": means,
            "stds": stds
        }, f)
    print("Model saved to logreg_model.pkl")



if __name__ == "__main__":
    main()