import sys
import math
import matplotlib.pyplot as plt
from describe import calculate_statistics, extract_numerical_columns
from utils import (
    read_csv_file,
    parse_csv_data,
    sigmoid,
    shuffle_data,
    normalization
)


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
def load_data(data_file):
    """Load and prepare training data."""
    lines = read_csv_file(data_file)
    headers, rows = parse_csv_data(lines)

    label_col = "Hogwarts House"
    numerical_columns = extract_numerical_columns(headers, rows)
    numeric_feature_names = sorted(numerical_columns.keys())

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

    # Normalize using utils function
    stats = calculate_statistics(numerical_columns)
    means = [stats[f]["mean"] for f in numeric_feature_names]
    stds = [stats[f]["std"] for f in numeric_feature_names]

    X_scaled = normalization(X, means, stds)

    return X_scaled, y


def train_all_houses(X, y_labels, train_func, **kwargs):
    """
    Train one-vs-rest classifiers for all houses using the given training function.

    Returns:
        dict: {house_name: losses_list}
    """
    houses = sorted(set(y_labels))
    results = {}

    for house in houses:
        # Create binary labels (1 for this house, 0 for others)
        y_binary = [1 if label == house else 0 for label in y_labels]

        # Train model
        _, _, losses = train_func(X, y_binary, **kwargs)
        results[house] = losses

    return results


def plot_technique(losses_dict, technique_name, epochs, save_path):
    """
    Create a plot showing loss curves for all houses for a single technique.

    Args:
        losses_dict: Dictionary mapping house names to loss lists
        technique_name: Name of the technique (for title)
        epochs: Number of epochs
        save_path: Path to save the plot
    """
    # House colors matching their official colors
    colors = {
        'Gryffindor': '#740001',
        'Hufflepuff': '#FFD700',
        'Ravenclaw': '#0E1A40',
        'Slytherin': '#1A472A'
    }

    plt.figure(figsize=(12, 7))

    for house, losses in sorted(losses_dict.items()):
        plt.plot(losses, label=house, color=colors.get(house, 'gray'),
                linewidth=2.5, alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{technique_name}\nTraining Loss by House', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, epochs - 1)

    # Add min/max loss info
    all_losses = [loss for losses in losses_dict.values() for loss in losses]
    plt.ylim(min(all_losses) * 0.95, max(all_losses) * 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python compare_training.py <dataset_path>")
        sys.exit(1)

    data_file = sys.argv[1]

    print("="*70)
    print("TRAINING COMPARISON: All Houses with Different Techniques")
    print("="*70)
    print(f"\nLoading data from {data_file}...")
    X, y_labels = load_data(data_file)

    print(f"Training on {len(X)} samples with {len(X[0])} features")
    print(f"Houses: {sorted(set(y_labels))}\n")

    # Training parameters
    epochs = 100
    learning_rate_batch = 0.5
    learning_rate_sgd = 0.1
    learning_rate_minibatch = 0.2
    batch_size = 32

    # Train with each technique for all houses
    print("-"*70)
    print("Training with Batch Gradient Descent...")
    print("-"*70)
    losses_batch_all = train_all_houses(
        X, y_labels, train_batch_gd,
        learning_rate=learning_rate_batch,
        epochs=epochs
    )
    print("  ✓ Batch GD complete")

    print("\n" + "-"*70)
    print("Training with Stochastic Gradient Descent...")
    print("-"*70)
    losses_sgd_all = train_all_houses(
        X, y_labels, train_sgd,
        learning_rate=learning_rate_sgd,
        epochs=epochs
    )
    print("  ✓ SGD complete")

    print("\n" + "-"*70)
    print("Training with Mini-Batch Gradient Descent...")
    print("-"*70)
    losses_minibatch_all = train_all_houses(
        X, y_labels, train_mini_batch_gd,
        learning_rate=learning_rate_minibatch,
        epochs=epochs,
        batch_size=batch_size
    )
    print("  ✓ Mini-Batch GD complete")

    # Create individual plots for each technique
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    import os
    os.makedirs('images', exist_ok=True)

    plot_technique(
        losses_batch_all,
        f'Batch Gradient Descent (lr={learning_rate_batch})',
        epochs,
        'images/training_batch_gd.png'
    )

    plot_technique(
        losses_sgd_all,
        f'Stochastic Gradient Descent (lr={learning_rate_sgd})',
        epochs,
        'images/training_sgd.png'
    )

    plot_technique(
        losses_minibatch_all,
        f'Mini-Batch Gradient Descent (lr={learning_rate_minibatch}, batch_size={batch_size})',
        epochs,
        'images/training_minibatch_gd.png'
    )

    # Print final losses for each technique and house
    print("\n" + "="*70)
    print("FINAL LOSS SUMMARY (after {} epochs)".format(epochs))
    print("="*70)

    print("\n1. Batch Gradient Descent:")
    for house in sorted(losses_batch_all.keys()):
        print(f"   {house:12s}: {losses_batch_all[house][-1]:.6f}")

    print("\n2. Stochastic Gradient Descent:")
    for house in sorted(losses_sgd_all.keys()):
        print(f"   {house:12s}: {losses_sgd_all[house][-1]:.6f}")

    print("\n3. Mini-Batch Gradient Descent:")
    for house in sorted(losses_minibatch_all.keys()):
        print(f"   {house:12s}: {losses_minibatch_all[house][-1]:.6f}")

    print("\n" + "="*70)
    print("✓ All plots saved to images/ directory")
    print("="*70)


if __name__ == "__main__":
    main()
