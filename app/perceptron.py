#the perceptron model

import numpy as np

class Perceptron:

    #Constructor for the perceptron class
    def __init__(self, learning_rate=0.01, max_iterations=1000, class_weight=None):

        self.learning_rate = learning_rate

        self.max_iterations = max_iterations

        self.weights = None

        self.bias = None

        self.class_weight = class_weight 
    
    #Training method

    def fit(self, X, Y):

        # Initialize weights and bias

        n_features = X.shape[1]

        self.weights = np.zeros(n_features)

        self.bias = 0

        if self.class_weight is None:

            n_pos = np.sum(Y == 1)

            n_neg = np.sum(Y == 0)

            self.class_weight = {0: 1.0, 1: np.sqrt(n_neg / n_pos)}

        
        # Training loop

        for iteration in range(self.max_iterations):

            errors = 0
            
            # Go through each training example

            for i in range(len(X)):

                # Calculate linear output

                linear_output = np.dot(X[i], self.weights) + self.bias
                
                # Make prediction (threshold at 0)

                prediction = 1 if linear_output >= 0 else 0
                
                # Update weights if prediction is wrong
                
                if prediction != Y[i]:

                    # Perceptron update rule
                    weight = self.class_weight[Y[i]]

                    update = self.learning_rate * weight * (Y[i] - prediction)

                    self.weights += update * X[i]

                    self.bias += update
                    
                    errors += 1

            if errors == 0:
                break
    
    def predict(self, X):
        
        linear_output = np.dot(X, self.weights) + self.bias

        return np.where(linear_output >= 0, 1, 0)    


def decision_score(self, X):

    return np.dot(X, self.weights) + self.bias

def predict(self, X):
    
    return np.where(self.decision_score(X) >= 0, 1, 0)