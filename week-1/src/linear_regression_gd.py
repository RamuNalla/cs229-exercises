import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Calculating Least mean Square cost function given training data and parameters
def lms_cost_function(x, y , theta_0, theta_1):    

    sum_error = (1/2)*np.sum(((theta_0 + theta_1*x) - y)**2)
    return sum_error


# Calculating Gradient of cost function parameters given training data and parameters (at that particular iteration)
def grad_cost_function(x, y, theta_0, theta_1):  

    grad_0 = np.sum((theta_0 + theta_1*x) - y)      # Note for derivative wrt theta_0, there will be no x (or as per Lecture-2, x_0 = 1)
    grad_1 = np.sum(((theta_0 + theta_1*x) - y)*x)

    return np.array([grad_0, grad_1])


np.random.seed(42)

x = np.linspace(0, 10, 100)                # An Array of x with values between 0 and 10
y = 2*x + 4 + np.random.normal(0, 1, 100)  # True function: y = 2x + 4 + noise   # Array

theta_0, theta_1 = 0, 0                    # Initial parametes

l_rate = 0.005                             # Learning rate
max_iters = 100                           # Maximum Iterations
tol = 1e-6                                 # Tolerance limit for gradient descent
m = len(x)

lms_array = []                             # Initial array to store all LMS calculations over iterations    
iterations_array = []                      # To store each iteration i

for i in range(max_iters):

    grad_theta = grad_cost_function(x, y, theta_0, theta_1)

    step_0 = l_rate*grad_theta[0]
    step_1 = l_rate*grad_theta[1]

    if np.abs(step_0) < tol and np.abs(step_1) < tol:    # Verify the step_0 and step_1 is greater than tolerance limit
        print(f"Converged at iteration {i}")
        break

    theta_0 -= step_0
    theta_1 -= step_1   

    lms = lms_cost_function(x, y, theta_0, theta_1)
    lms_array.append(lms)
    iterations_array.append(i) 


print(f"Estimated parameters: θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")
print("True values: θ₀ = 4, θ₁ = 2")