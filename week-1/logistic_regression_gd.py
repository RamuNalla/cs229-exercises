import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# SIGMOID FUNCTION
def sigmoid (x):
    return 1/(1+np.exp(-x))

# LOSS FUNCTION IN LOGISTIC REGRESSION
def log_loss_func(x, y, theta):
    n = x.shape[0]
    g_x = x @ theta        # x is a 2D data and theta is a vector with [theta_0, theta_1, theta_2]
    h_x = sigmoid(g_x)
    epsilon = 1e-15        # avoid log(0)
    loss_function = np.sum(y * np.log(h_x + epsilon) + (1-y) * np.log(1 - h_x + epsilon))/n

    return loss_function

# GREADIENT OF THE LOSS FUNCTION
def grad_loss_function(x, y, theta):  
    n = x.shape[0]
    g_x = x @ theta        # x is a 2D data and theta is a vector with [theta_0, theta_1, theta_2]
    h_x = sigmoid(g_x)

    # grad_0 = np.sum(h_x - y) / n      # Note for derivative wrt theta_0, there will be no x (or as per Lecture-2, x_0 = 1)
    # grad_1 = np.sum((h_x - y)*x) / n

    grad = (1/n)*(x.T @ (h_x - y))    # gradient will be [x1, x2, x3]*[h_x-y] ==> grad_1 = sum of (x1*(h_x - y))  Making x transpose and dot product with h_x - y yield same result.  Note grad is a vector with [grad_0, grad_1, grad_2]

    return grad

# BATCH GRADIENT ASCENT FOR EACH LEARNING RATE
def batch_gradient_ascent(x, y, learning_rate, max_iters, tol):
    
    theta = np.zeros((x.shape[1],1))      # Initialization, theta to have 1 column

    n = x.shape[0]
    log_loss_array = []
    iterations_array = []

    for i in range(max_iters):
        grad_theta = grad_loss_function(x, y, theta)

        # step_0 = learning_rate * grad_theta[0]
        # step_1 = learning_rate * grad_theta[1]
        steps = learning_rate * grad_theta

        # if np.abs(np.sum(steps)/len(steps)) < tol:              ## Or np.linalg.norm(steps)
        #     print(f"Converged at iteration {i} for learning rate {learning_rate}")
        #     break
        steps = steps.reshape(-1, 1)        # Steps is a 2D vector now with size (3,1) earlier it is an array with (3,)
        theta += steps        #  Gradient Ascent

        log_loss = log_loss_func(x, y, theta)
        log_loss_array.append(log_loss)
        iterations_array.append(i)

    return theta, log_loss_array, iterations_array, i


# ---------- Greadient Ascent Implementation ------------------------------------------------

X, y = make_blobs(n_samples = 200, centers = 2, n_features = 2, random_state = 42)   # 200 samples with 2 classes in the data. Total number of features are 2 meaning it is a 2D data, therefore theta vector will have theta_0, theta_1, theta_2
X = np.c_[np.ones((X.shape[0], 1)), X]   #horizontally concatenates two arrays (columns), first one with all ones
y = y.reshape(-1, 1)

max_iters = 10000                          # Maximum Iterations
tol = 1e-4                                 # Tolerance limit for gradient descent
l_rates = [0.001, 0.005, 0.01, 0.05]             # Learning rates


for lr in l_rates:
    
    theta, log_loss_array, iterations_array, iters = batch_gradient_ascent(X, y, lr, max_iters, tol)

    print(f"Learning Rate: {lr}")
    print(f"  θ₀ = {theta[0][0]:.3f}, θ₁ = {theta[1][0]:.3f}, θ2 = {theta[2][0]:.3f}")
    print(f"  Iterations: {iters}")
    

