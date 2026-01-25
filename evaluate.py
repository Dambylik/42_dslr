"""
Put these files in the same folder as `houses.csv` and `dataset_truth.csv`.

Usage:
    $ python evaluate.py
"""
from __future__ import print_function
import csv
import sys
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_csv(filename):
    """Load a CSV file and return a list with datas (corresponding to truths or
    predictions).
    """
    datas = list()
    with open(filename, 'r') as opened_csv:
        read_csv = csv.reader(opened_csv, delimiter=',')
        for line in read_csv:
            datas.append(line[1])
    # Clean the header cell
    datas.remove("Hogwarts House")
    return datas


def plot_confusion_matrix(y_true, y_pred, save_path='images/confusion_matrix.png'):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    houses = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=houses)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=houses, yticklabels=houses)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True House', fontsize=12)
    plt.xlabel('Predicted House', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_accuracy_by_house(y_true, y_pred, save_path='images/accuracy_by_house.png'):
    """
    Plot accuracy for each house.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    houses = sorted(set(y_true))
    accuracies = []

    for house in houses:
        # Get indices where true label is this house
        indices = [i for i, label in enumerate(y_true) if label == house]
        if len(indices) > 0:
            correct = sum(1 for i in indices if y_pred[i] == house)
            accuracy = correct / len(indices)
        else:
            accuracy = 0.0
        accuracies.append(accuracy * 100)

    plt.figure(figsize=(10, 6))
    colors = ['#740001', '#FFD700', '#0E1A40', '#1A472A']  # Gryffindor, Hufflepuff, Ravenclaw, Slytherin
    bars = plt.bar(houses, accuracies, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.title('Accuracy by Hogwarts House', fontsize=16, fontweight='bold')
    plt.xlabel('House', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy by house saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    if os.path.isfile("./datasets/dataset_truth.csv"):
        truths = load_csv("./datasets/dataset_truth.csv")
    else:
        sys.exit("Error: missing dataset_truth.csv in the current directory.")
    if os.path.isfile("houses.csv"):
        predictions = load_csv("houses.csv")
    else:
        sys.exit("Error: missing houses.csv in the current directory.")

    # Verify lengths match
    if len(truths) != len(predictions):
        sys.exit("Error: truths and predictions have different lengths.")

    # Calculate accuracy using scikit-learn
    score = accuracy_score(truths, predictions)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Your score on test set: {score:.3f} ({score*100:.2f}%)")
    print(f"Total samples: {len(truths)}")
    print(f"Correct predictions: {int(score * len(truths))}")
    print(f"Incorrect predictions: {len(truths) - int(score * len(truths))}")

    if score >= .98:
        print("\nGood job! Mc Gonagall congratulates you.")
    else:
        print("\nToo bad, Mc Gonagall flunked you.")

    # Print detailed classification report
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT")
    print("-"*60)
    print(classification_report(truths, predictions))

    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)

    # Generate plots
    print("\n" + "-"*60)
    print("GENERATING VISUALIZATIONS")
    print("-"*60)
    plot_confusion_matrix(truths, predictions)
    plot_accuracy_by_house(truths, predictions)

    print("\nEvaluation complete!")
