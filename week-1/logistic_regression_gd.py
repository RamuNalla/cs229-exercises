import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# SIGMOID FUNCTION
def sigmoid (x):
    
    return np.where(                                   ## To make it numerically stable, when x = -1000, exp(1000) is beyond float 64 limit
        x >= 0,                                        ## np.where is vectorized, it applies the logic element-wise over arrays
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )                                                  ## Simple if-else condition is can be used for scalar, Here the sigmoid function inputs are vectorized (numpy arrays), therefore using np.where


# LOSS FUNCTION IN LOGISTIC REGRESSION
def log_loss_func(x, y, theta):
    n = x.shape[0]
    g_x = x @ theta        # x is a 2D data and theta is a vector with [theta_0, theta_1, theta_2]
    h_x = sigmoid(g_x)
    epsilon = 1e-15        # avoid log(0)
    loss_function = -np.sum(y * np.log(h_x + epsilon) + (1-y) * np.log(1 - h_x + epsilon))/n

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

# BATCH GRADIENT DESCENT FOR EACH LEARNING RATE
def batch_gradient_descent(x, y, learning_rate, max_iters, tol):
    
    theta = np.zeros((x.shape[1],1))      # Initialization, theta to have 1 column

    n = x.shape[0]
    log_loss_array = []
    iterations_array = []

    for i in range(max_iters):
        grad_theta = grad_loss_function(x, y, theta)

        # step_0 = learning_rate * grad_theta[0]
        # step_1 = learning_rate * grad_theta[1]
        steps = learning_rate * grad_theta
        log_loss = log_loss_func(x, y, theta)
        if i>50 and np.abs(log_loss_array[-1] - log_loss) < tol:              ## We can alose use np.sum(steps)/len(steps), but the steps could cancel each other, therefore norm is required
            print(f"Converged at iteration {i} for learning rate {learning_rate}")
            break
        steps = steps.reshape(-1, 1)        # Steps is a 2D vector now with size (3,1) earlier it is an array with (3,)
        theta -= steps        #  Gradient Ascent

        
        log_loss_array.append(log_loss)
        iterations_array.append(i)

    return theta, log_loss_array, iterations_array, i


# ---------- Greadient Descent Implementation ------------------------------------------------

X, y = make_blobs(n_samples = 200, centers = 2, n_features = 2, random_state = 42)   # 200 samples with 2 classes in the data. Total number of features are 2 meaning it is a 2D data, therefore theta vector will have theta_0, theta_1, theta_2
X = np.c_[np.ones((X.shape[0], 1)), X]   #horizontally concatenates two arrays (columns), first one with all ones
y = y.reshape(-1, 1)                     # y is a column vector

max_iters = 80000                          # Maximum Iterations
tol = 1e-6                              # Tolerance limit for gradient descent
l_rates = [0.001, 0.01, 0.05, 0.08, 0.1]             # Learning rates
colors = ['red', 'green', 'blue', 'grey', 'black']

fig, ax = plt.subplots(1, 2, figsize=(14, 6))



for lr, color in zip(l_rates, colors):
    
    theta, log_loss_array, iterations_array, iters = batch_gradient_descent(X, y, lr, max_iters, tol)

    print(f"Learning Rate: {lr}")
    print(f"  θ₀ = {theta[0][0]:.3f}, θ₁ = {theta[1][0]:.3f}, θ2 = {theta[2][0]:.3f}")
    print(f"  Iterations: {iters}\n")

    # Plotting the log loss array with number of iterations
    ax[0].plot(iterations_array, log_loss_array, color=color, label=f"LR={lr}")

    #Plotting the decision boundary for each learning rate
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1            # Identifying the minimum and maximum values from the data
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),           # Creating a mesh grid between the minimum and maximum. xx will be 200 x 200 2D matrix
                         np.linspace(y_min, y_max, 200))
                                                                              # xx.ravel() flattens the 2D matrix into 1d vector with length 40000
    grid = np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()]   # Column stacking and therfore creating a grid with [1,x,y] value of all 200x200 points
    probs = sigmoid(grid @ theta).reshape(xx.shape)                           # predicting the probability with passing the theta x point into sigmoid function

    # Plot decision boundary (probability = 0.5)
    ax[1].contour(xx, yy, probs, levels=[0.5], colors=color, linestyles='solid', linewidths=2)   # It draws lines on (xx, yy) where probs exactly equals 0.5


ax[0].set_xlabel("Iterations")
ax[0].set_ylabel("Log Loss")
ax[0].set_title(f"Log Loss Curve for Learning Rate {lr}")
ax[0].legend()

ax[1].scatter(X[:, 1][y[:,0] ==0], X[:, 2][y[:,0] == 0], color='blue', label='Class 0')    # Plotting the all x1, x2 where y=0 with blue dots
ax[1].scatter(X[:, 1][y[:,0] ==1], X[:, 2][y[:,0] == 1], color='red', label='Class 1')     # Plotting all x1, x2 where y=1 with red dots

ax[1].set_xlabel("Feature 1")
ax[1].set_ylabel("Feature 2")
ax[1].set_title("Decision Boundaries for Different Learning Rates")
ax[1].legend()



plt.show()
    


