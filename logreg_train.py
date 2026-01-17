import sys
import math

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

        

def main():
    """Main programm"""
        
    print("Logistic Regression project")
    print("---------------------------")
    print(sigmoid(-5))
    print(log_loss(1, 0.9))
    

if __name__ == "__main__":
    main()