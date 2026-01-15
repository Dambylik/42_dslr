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



def main():
    """Main programm"""
        
    print("Logistic Regression project")
    print("---------------------------")
    print(sigmoid(-5))
    print(log_loss(1, 0.9))
    

if __name__ == "__main__":
    main()