import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import read_csv_file, parse_csv_data
from histogram import extract_course_by_house
from scatter_plot import extract_pair_by_house

def generate_histogram():
    """Generate histogram image."""
    lines = read_csv_file("datasets/dataset_train.csv")
    headers, rows = parse_csv_data(lines)

    course_name = "Care of Magical Creatures"
    houses_data = extract_course_by_house(rows, headers, course_name)

    plt.figure(figsize=(10, 6))
    colors = {'Gryffindor': '#740001', 'Ravenclaw': '#0E1A40',
              'Hufflepuff': '#FFD800', 'Slytherin': '#1A472A'}

    for house, grades in houses_data.items():
        if not grades:
            continue
        plt.hist(grades, bins=20, alpha=0.6, label=house, color=colors.get(house))

    plt.title(f"Score Distribution: {course_name}\n(Most homogeneous across houses)")
    plt.xlabel("Score")
    plt.ylabel("Number of Students")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: images/histogram.png")


def generate_scatter():
    """Generate scatter plot image."""
    lines = read_csv_file("datasets/dataset_train.csv")
    headers, rows = parse_csv_data(lines)

    feature_x = "Astronomy"
    feature_y = "Defense Against the Dark Arts"
    houses_data = extract_pair_by_house(rows, headers, feature_x, feature_y)

    plt.figure(figsize=(10, 6))
    colors = {'Gryffindor': '#740001', 'Ravenclaw': '#0E1A40',
              'Hufflepuff': '#FFD800', 'Slytherin': '#1A472A'}

    for house, values in houses_data.items():
        if not values["x"]:
            continue
        plt.scatter(values["x"], values["y"], alpha=0.6, s=15,
                   label=house, color=colors.get(house))

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{feature_x} vs {feature_y}\n(Correlation r = -1.0, most similar features)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('images/scatter_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: images/scatter_plot.png")


def generate_pair_plot():
    """Generate pair plot image."""
    df = pd.read_csv("datasets/dataset_train.csv")
    drop_cols = ["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
    df = df.drop(columns=drop_cols)

    # Select subset of features for cleaner visualization
    features = ["Hogwarts House", "Astronomy", "Herbology", "Charms", "Flying"]
    df_subset = df[features]

    palette = {'Gryffindor': '#740001', 'Ravenclaw': '#0E1A40',
               'Hufflepuff': '#FFD800', 'Slytherin': '#1A472A'}

    g = sns.pairplot(df_subset, hue="Hogwarts House", diag_kind="hist",
                     plot_kws={"alpha": 0.6, "s": 15}, palette=palette)
    g.fig.suptitle("Pair Plot (Feature Relationships by House)", y=1.02)
    plt.savefig('images/pair_plot.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Generated: images/pair_plot.png")


def generate_training_comparison():
    """Generate training comparison image (copy from compare_training.py logic)."""
    import math
    import random
    from describe import calculate_statistics, extract_numerical_columns

    def sigmoid(z):
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            return exp_z / (1 + exp_z)

    def compute_loss(X, y, w, b):
        m = len(X)
        total_loss = 0.0
        eps = 1e-15
        for i in range(m):
            z = sum(w[j] * X[i][j] for j in range(len(w))) + b
            y_hat = sigmoid(z)
            y_hat = max(eps, min(1 - eps, y_hat))
            total_loss += -(y[i] * math.log(y_hat) + (1 - y[i]) * math.log(1 - y_hat))
        return total_loss / m

    def shuffle_data(X, y):
        n = len(X)
        X_s, y_s = X[:], y[:]
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            X_s[i], X_s[j] = X_s[j], X_s[i]
            y_s[i], y_s[j] = y_s[j], y_s[i]
        return X_s, y_s

    def train_batch(X, y, lr, epochs):
        n_features, m = len(X[0]), len(X)
        w, b = [0.0] * n_features, 0.0
        losses = []
        for _ in range(epochs):
            dw, db = [0.0] * n_features, 0.0
            for i in range(m):
                z = sum(w[j] * X[i][j] for j in range(n_features)) + b
                error = sigmoid(z) - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                w[j] -= lr * dw[j] / m
            b -= lr * db / m
            losses.append(compute_loss(X, y, w, b))
        return losses

    def train_sgd(X, y, lr, epochs):
        n_features = len(X[0])
        w, b = [0.0] * n_features, 0.0
        losses = []
        for _ in range(epochs):
            X_s, y_s = shuffle_data(X, y)
            for i in range(len(X_s)):
                z = sum(w[j] * X_s[i][j] for j in range(n_features)) + b
                error = sigmoid(z) - y_s[i]
                for j in range(n_features):
                    w[j] -= lr * error * X_s[i][j]
                b -= lr * error
            losses.append(compute_loss(X, y, w, b))
        return losses

    def train_minibatch(X, y, lr, epochs, batch_size=32):
        n_features, n_samples = len(X[0]), len(X)
        w, b = [0.0] * n_features, 0.0
        losses = []
        for _ in range(epochs):
            X_s, y_s = shuffle_data(X, y)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_b, y_b = X_s[start:end], y_s[start:end]
                m = len(X_b)
                dw, db = [0.0] * n_features, 0.0
                for i in range(m):
                    z = sum(w[j] * X_b[i][j] for j in range(n_features)) + b
                    error = sigmoid(z) - y_b[i]
                    for j in range(n_features):
                        dw[j] += error * X_b[i][j]
                    db += error
                for j in range(n_features):
                    w[j] -= lr * dw[j] / m
                b -= lr * db / m
            losses.append(compute_loss(X, y, w, b))
        return losses

    # Load data
    lines = read_csv_file("datasets/dataset_train.csv")
    headers, rows = parse_csv_data(lines)
    numerical_columns = extract_numerical_columns(headers, rows)
    numeric_features = list(numerical_columns.keys())

    X, y_labels = [], []
    for row in rows:
        try:
            features = [float(row[headers.index(f)]) if row[headers.index(f)].strip() != '' else float('nan') for f in numeric_features]
            if any(str(x) == 'nan' for x in features):
                continue
            X.append(features)
            y_labels.append(row[headers.index("Hogwarts House")])
        except:
            continue

    stats = calculate_statistics(numerical_columns)
    means = [stats[f]["mean"] for f in numeric_features]
    stds = [stats[f]["std"] for f in numeric_features]
    X_scaled = [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in X]
    y = [1 if label == "Gryffindor" else 0 for label in y_labels]

    epochs = 100
    losses_batch = train_batch(X_scaled, y, 0.5, epochs)
    losses_sgd = train_sgd(X_scaled, y, 0.1, epochs)
    losses_mini = train_minibatch(X_scaled, y, 0.2, epochs, 32)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses_batch, label='Batch GD (lr=0.5)', linewidth=2)
    ax.plot(losses_sgd, label='SGD (lr=0.1)', linewidth=2, alpha=0.8)
    ax.plot(losses_mini, label='Mini-Batch GD (lr=0.2, batch=32)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss: Gradient Descent Methods Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('images/training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: images/training_comparison.png")


def generate_logistic_regression_diagram():
    """Generate a diagram explaining logistic regression."""
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sigmoid function
    ax1 = axes[0]
    x = np.linspace(-6, 6, 100)
    y = 1 / (1 + np.exp(-x))
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision boundary (0.5)')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_xlabel('z = w·x + b')
    ax1.set_ylabel('σ(z) = Probability')
    ax1.set_title('Sigmoid Function: σ(z) = 1/(1+e⁻ᶻ)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # One-vs-Rest visualization
    ax2 = axes[1]
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    colors = ['#740001', '#FFD800', '#0E1A40', '#1A472A']
    y_pos = [0.8, 0.6, 0.4, 0.2]

    for i, (house, color, y) in enumerate(zip(houses, colors, y_pos)):
        ax2.barh(y, 0.7, height=0.15, color=color, alpha=0.8, label=house)
        ax2.text(0.75, y, f'Classifier {i+1}: {house} vs Rest', va='center', fontsize=10)

    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 1)
    ax2.set_title('One-vs-Rest (OvR) Multi-class Strategy')
    ax2.set_xlabel('Each classifier outputs P(house)')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('images/logistic_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: images/logistic_regression.png")


if __name__ == "__main__":
    import os
    os.makedirs("images", exist_ok=True)

    print("Generating visualization images...")
    generate_histogram()
    generate_scatter()
    generate_pair_plot()
    generate_training_comparison()
    generate_logistic_regression_diagram()
    print("\nAll images generated in 'images/' folder!")
