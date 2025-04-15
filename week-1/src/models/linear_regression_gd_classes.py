import numpy as np

class LinearRegressionGD:                                                 # Defining a class LinearRegressionGD
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-6):     # The class require these inputs
        self.lr = learning_rate                                           # Instance attributes declaration. Note that parameters are declared as instance attributes
        self.max_iters = max_iters
        self.tol = tol
        self.theta_0 = 0
        self.theta_1 = 0
        self.cost_history = []

    def cost(self, X, y):                                                 # Cost function for a given thetas
        n = len(X)
        pred = self.theta_0 + self.theta_1 * X
        return (1/(2*n))* np.sum((pred-y)**2)
    
    def gradient(self, X, y):                                             # Gradient function for a given thetas
        n=len(X)
        pred = self.theta_0 + self.theta_1 * X
        error = pred - y
        grad_0 = np.sum(error)/n
        grad_1 = np.sum(error * X)/n
        return grad_0, grad_1
    
    def fit(self, X, y):                                                  # Running the gradient descent till convergence
        for i in range(self.max_iters):

            grad_0, grad_1 = self.gradient(X, y)
            step_0 = self.lr * grad_0
            step_1 = self.lr * grad_1
            
            if np.abs(step_0) < self.tol and np.abs(step_1) < self.tol:
                print(f"Converged at iteration {i}")
                break

            self.theta_0 -= step_0
            self.theta_1 -= step_1

            current_cost = self.cost(X, y)
            self.cost_history.append(current_cost)

    def predict(self, X_new):                               
        return self.theta_0 + self.theta_1*X_new                           # Predicting the new Ys
    
    def get_params(self):                                                  # If the user wants to know the parameteres, this will return it
        return self.theta_0, self.theta_1

        