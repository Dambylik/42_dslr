"""
Visualization comparing different gradient descent training techniques.
Usage: python compare_training.py
"""
import math
import random
import matplotlib.pyplot as plt
from describe import calculate_statistics, extract_numerical_columns
from utils import read_csv_file, parse_csv_data


# =====================
# Math utilities
# =====================
def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)


def compute_loss(X, y, w, b):
    """Compute average log loss over all samples."""
    m = len(X)
    total_loss = 0.0
    eps = 1e-15
    for i in range(m):
        z = sum(w[j] * X[i][j] for j in range(len(w))) + b
        y_hat = sigmoid(z)
        y_hat = max(eps, min(1 - eps, y_hat))
        total_loss += -(y[i] * math.log(y_hat) + (1 - y[i]) * math.log(1 - y_hat))
    return total_loss / m


def compute_accuracy(X, y, w, b):
    """Compute accuracy on dataset."""
    correct = 0
    for i in range(len(X)):
        z = sum(w[j] * X[i][j] for j in range(len(w))) + b
        pred = 1 if sigmoid(z) >= 0.5 else 0
        if pred == y[i]:
            correct += 1
    return correct / len(X)


def shuffle_data(X, y):
    n = len(X)
    X_shuffled = X[:]
    y_shuffled = y[:]
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        X_shuffled[i], X_shuffled[j] = X_shuffled[j], X_shuffled[i]
        y_shuffled[i], y_shuffled[j] = y_shuffled[j], y_shuffled[i]
    return X_shuffled, y_shuffled


# =====================
# Training methods with loss tracking
# =====================
def train_batch_gd(X, y, learning_rate, epochs):
    """Batch Gradient Descent with loss tracking."""
    n_features = len(X[0])
    m = len(X)
    w = [0.0] * n_features
    b = 0.0
    losses = []

    for epoch in range(epochs):
        # Compute gradients over all samples
        dw = [0.0] * n_features
        db = 0.0
        for i in range(m):
            z = sum(w[j] * X[i][j] for j in range(n_features)) + b
            y_hat = sigmoid(z)
            error = y_hat - y[i]
            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error
        dw = [g / m for g in dw]
        db /= m

        # Update
        for j in range(n_features):
            w[j] -= learning_rate * dw[j]
        b -= learning_rate * db

        # Track loss
        losses.append(compute_loss(X, y, w, b))

    return w, b, losses


def train_sgd(X, y, learning_rate, epochs):
    """Stochastic Gradient Descent with loss tracking."""
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
    losses = []

    for epoch in range(epochs):
        X_shuffled, y_shuffled = shuffle_data(X, y)
        for i in range(len(X_shuffled)):
            z = sum(w[j] * X_shuffled[i][j] for j in range(n_features)) + b
            y_hat = sigmoid(z)
            error = y_hat - y_shuffled[i]
            for j in range(n_features):
                w[j] -= learning_rate * error * X_shuffled[i][j]
            b -= learning_rate * error

        # Track loss at end of epoch
        losses.append(compute_loss(X, y, w, b))

    return w, b, losses


def train_mini_batch_gd(X, y, learning_rate, epochs, batch_size=32):
    """Mini-Batch Gradient Descent with loss tracking."""
    n_features = len(X[0])
    n_samples = len(X)
    w = [0.0] * n_features
    b = 0.0
    losses = []

    for epoch in range(epochs):
        X_shuffled, y_shuffled = shuffle_data(X, y)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            m = len(X_batch)

            dw = [0.0] * n_features
            db = 0.0
            for i in range(m):
                z = sum(w[j] * X_batch[i][j] for j in range(n_features)) + b
                y_hat = sigmoid(z)
                error = y_hat - y_batch[i]
                for j in range(n_features):
                    dw[j] += error * X_batch[i][j]
                db += error
            dw = [g / m for g in dw]
            db /= m

            for j in range(n_features):
                w[j] -= learning_rate * dw[j]
            b -= learning_rate * db

        # Track loss at end of epoch
        losses.append(compute_loss(X, y, w, b))

    return w, b, losses


# =====================
# Data loading
# =====================
def load_data():
    """Load and prepare training data."""
    lines = read_csv_file("datasets/dataset_train.csv")
    headers, rows = parse_csv_data(lines)

    label_col = "Hogwarts House"
    numerical_columns = extract_numerical_columns(headers, rows)
    numeric_feature_names = list(numerical_columns.keys())

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

    # Normalize
    stats = calculate_statistics(numerical_columns)
    means = [stats[f]["mean"] for f in numeric_feature_names]
    stds = [stats[f]["std"] for f in numeric_feature_names]

    X_scaled = []
    for row in X:
        scaled_row = [(row[j] - means[j]) / stds[j] for j in range(len(row))]
        X_scaled.append(scaled_row)

    return X_scaled, y


def main():
    print("Loading data...")
    X, y_labels = load_data()

    # Use Gryffindor as binary classification target
    y = [1 if label == "Gryffindor" else 0 for label in y_labels]

    print(f"Training on {len(X)} samples with {len(X[0])} features")
    print("Training with different methods...\n")

    # Train with each method (using comparable total iterations)
    epochs = 100

    print("Training Batch GD...")
    _, _, losses_batch = train_batch_gd(X, y, learning_rate=0.5, epochs=epochs)

    print("Training SGD...")
    _, _, losses_sgd = train_sgd(X, y, learning_rate=0.1, epochs=epochs)

    print("Training Mini-Batch GD (batch_size=32)...")
    _, _, losses_mini32 = train_mini_batch_gd(X, y, learning_rate=0.2, epochs=epochs, batch_size=32)

    print("Training Mini-Batch GD (batch_size=64)...")
    _, _, losses_mini64 = train_mini_batch_gd(X, y, learning_rate=0.3, epochs=epochs, batch_size=64)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss over epochs
    ax1 = axes[0]
    ax1.plot(losses_batch, label='Batch GD (lr=0.5)', linewidth=2)
    ax1.plot(losses_sgd, label='SGD (lr=0.1)', linewidth=2, alpha=0.8)
    ax1.plot(losses_mini32, label='Mini-Batch 32 (lr=0.2)', linewidth=2)
    ax1.plot(losses_mini64, label='Mini-Batch 64 (lr=0.3)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, epochs)

    # Plot 2: Zoomed view of convergence (last 50 epochs)
    ax2 = axes[1]
    start = max(0, epochs - 50)
    ax2.plot(range(start, epochs), losses_batch[start:], label='Batch GD', linewidth=2)
    ax2.plot(range(start, epochs), losses_sgd[start:], label='SGD', linewidth=2, alpha=0.8)
    ax2.plot(range(start, epochs), losses_mini32[start:], label='Mini-Batch 32', linewidth=2)
    ax2.plot(range(start, epochs), losses_mini64[start:], label='Mini-Batch 64', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Convergence (Last 50 Epochs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print final losses
    print("\n" + "="*50)
    print("Final Loss after {} epochs:".format(epochs))
    print("="*50)
    print(f"  Batch GD:       {losses_batch[-1]:.6f}")
    print(f"  SGD:            {losses_sgd[-1]:.6f}")
    print(f"  Mini-Batch 32:  {losses_mini32[-1]:.6f}")
    print(f"  Mini-Batch 64:  {losses_mini64[-1]:.6f}")

    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to training_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
