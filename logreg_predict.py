import sys
from utils import (
    read_csv_file,
    parse_csv_data,
    sigmoid,
    normalize,
    load_model
)


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
    models, means, stds, feature_names = load_model(model_path)

    lines = read_csv_file(data_path)
    headers, rows = parse_csv_data(lines)

    # Map feature indices
    feature_indices = [headers.index(f) for f in feature_names]

    predictions = []

    for row in rows:
        x = []
        for i, idx in enumerate(feature_indices):
            value = row[idx]
            # Use mean imputation for missing values instead of 0.0
            # This way, after normalization, missing values become 0.0 (neutral)
            if value != "":
                x.append(float(value))
            else:
                x.append(means[i])  # Fill with mean value

        x_norm = normalize(x, means, stds)
        house = predict_house(x_norm, models)
        predictions.append(house)

    with open("houses.csv", "w") as f:
        f.write("Index,Hogwarts House\n")
        for idx, h in enumerate(predictions):
            f.write(f"{idx},{h}\n")


if __name__ == "__main__":
    main()
