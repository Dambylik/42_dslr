import sys
import json
import math
from utils import read_csv_file, parse_csv_data


def sigmoid(z):
    """Numerically stable sigmoid function"""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)


def normalize(x, means, stds):
    return [(x[i] - means[i]) / stds[i] for i in range(len(x))]


def predict_house(x, models):
    """Predict house for one student usoing OVR model"""
    best_house = None
    best_prob = -1

    for house, params in models.items():
        w = params["weights"]
        b = params["bias"]
        z = sum(w[i] * x[i] for i in range(len(w))) + b
        prob = sigmoid(z)
        if prob > best_prob:
            best_prob = prob
            best_house = house
    return best_house


def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py model.json dataset_test.csv")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    # Load model
    with open(model_path, "r") as f:
        model = json.load(f)

    models = model["models"]
    means = model["means"]
    stds = model["stds"]
    feature_names = model["features"]

    # Read test data
    lines = read_csv_file(data_path)
    headers, rows = parse_csv_data(lines)

    # Map feature indices
    feature_indices = [headers.index(f) for f in feature_names]

    predictions = []

    for row in rows:
        x = []
        for idx in feature_indices:
            value = row[idx]
            x.append(float(value) if value != "" else 0.0)

        x_norm = normalize(x, means, stds)
        house = predict_house(x_norm, models)
        predictions.append(house)

    # Write output with index and header
    with open("houses.csv", "w") as f:
        f.write("Index,Hogwarts House\n")
        for idx, h in enumerate(predictions):
            f.write(f"{idx},{h}\n")


if __name__ == "__main__":
    main()
