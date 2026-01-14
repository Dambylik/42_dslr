import sys
import math

def sigmoid(z):
    """sigmoid function : σ(z) = 1 / (1 + e⁻ᶻ)"""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)



def main():
    """Main programm"""
        
    print("Logistic Regression project")
    print("---------------------------")
    print(sigmoid(-5))

    

if __name__ == "__main__":
    main()