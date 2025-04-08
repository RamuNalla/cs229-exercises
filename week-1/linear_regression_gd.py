import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

# ---------- Greadient Descent Implementation -----------------------------------------


# Calculating Least mean Square cost function given training data and parameters
def lms_cost_function(x, y , theta_0, theta_1, n):    

    sum_error = (1/2)*np.sum(((theta_0 + theta_1*x) - y)**2)
    return sum_error/n


# Calculating Gradient of cost function parameters given training data and parameters (at that particular iteration)
def grad_cost_function(x, y, theta_0, theta_1, n):  

    grad_0 = np.sum((theta_0 + theta_1*x) - y) / n     # Note for derivative wrt theta_0, there will be no x (or as per Lecture-2, x_0 = 1)
    grad_1 = np.sum(((theta_0 + theta_1*x) - y)*x) / n

    return np.array([grad_0, grad_1])


# Perform Batch Gradient Descent for each learning rate
def batch_gradient_descent(x, y, learning_rate, max_iters, tol):
    theta_0, theta_1 = 0, 0
    n = len(x)
    lms_array = []
    iterations_array = []

    for i in range(max_iters):
        grad_theta = grad_cost_function(x, y, theta_0, theta_1, n)
        step_0 = learning_rate * grad_theta[0]
        step_1 = learning_rate * grad_theta[1]

        if np.abs(step_0) < tol and np.abs(step_1) < tol:
            print(f"Converged at iteration {i} for learning rate {learning_rate}")
            break

        theta_0 -= step_0
        theta_1 -= step_1

        lms = lms_cost_function(x, y, theta_0, theta_1, n)
        lms_array.append(lms)
        iterations_array.append(i)

    return theta_0, theta_1, lms_array, iterations_array, i



np.random.seed(42)

x = np.linspace(0, 10, 100)                # An Array of x with values between 0 and 10
y = 2*x + 4 + np.random.normal(0, 1, 100)  # True function: y = 2x + 4 + noise   # Array

max_iters = 10000                          # Maximum Iterations
tol = 1e-6                                 # Tolerance limit for gradient descent
l_rates = [0.001, 0.005, 0.01, 0.05]             # Learning rates
colors = ['red', 'green', 'blue', 'grey']

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(x, y, label="Data", color="black", alpha=0.6)


# Perform batch gradient descent for each learning rate and the plot
for lr, color in zip(l_rates, colors):
    
    theta_0, theta_1, lms_array, iterations_array, iters = batch_gradient_descent(x, y, lr, max_iters, tol)

    print(f"Learning Rate: {lr}")
    print(f"  θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")
    print(f"  Iterations: {iters}")
    
    # R-squared Computation for each learning rate
    y_pred = theta_0 + theta_1 * x
    SS_res = np.sum((y - y_pred)**2)
    SS_tot = np.sum((y - np.mean(y))**2)
    R_squared = 1 - (SS_res / SS_tot)
    print(f"  R²: {R_squared:.4f}\n")

    # Plot regression line for each learning rate
    ax[0].plot(x, y_pred, color=color, label=f"LR={lr}")

    # Plot cost function for each learning rate
    ax[1].plot(iterations_array, lms_array, color=color, label=f"LR={lr}")

ax[0].set_title("Regression Lines for Different Learning Rates")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].legend()

ax[1].set_title("Cost Function vs Iterations")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Cost (log scale)")
ax[1].set_yscale("log")
ax[1].legend()



# ---------- Normal Equation Implementation ----------------------------------

def normal_equations(x, y):
    X = np.column_stack((np.ones_like(x), x))   ## Added intercept column x_0

    xt_x = X.T @ X   
    xt_y = X.T @ y
    theta_normal_equations = np.linalg.inv(xt_x) @ xt_y

    return theta_normal_equations


theta_normal_equations = normal_equations(x, y)
print(f"Parameters from normal equations:")
print(f"  θ₀ = {theta_normal_equations[0]:.3f}, θ₁ = {theta_normal_equations[1]:.3f}\n")




# ---------- Sklearn Linear Regression ----------------------------------------

model = LinearRegression()
x_reshaped = x.reshape(-1,1)  ## Converts into a 2D array of size (100,1)
model.fit(x_reshaped, y)

joblib.dump(model, "week-1/saved_models/linear_regression_model.joblib")

theta_0_sklearn = model.intercept_
theta_1_sklearn = model.coef_[0]

print(f"Parameters from sklearn Linear Regression:")
print(f"  θ₀ = {theta_0_sklearn:.3f}, θ₁ = {theta_1_sklearn:.3f}")

plt.show()