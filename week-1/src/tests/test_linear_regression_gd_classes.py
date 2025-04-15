import numpy as np
import matplotlib.pyplot as plt
from models.linear_regression_gd_classes import LinearRegressionGD

np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 4 + np.random.normal(0, 1, 100)

model = LinearRegressionGD(learning_rate=0.01, max_iters=10000)
model.fit(X, y)

theta_0, theta_1 = model.get_params()
print(f"θ₀ = {theta_0:.3f}, θ₁ = {theta_1:.3f}")

y_pred = model.predict(X)

plt.scatter(X, y, label="Data", color="black", alpha=0.6)
plt.plot(X, y_pred, color="blue", label="Prediction")
plt.legend()
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Plot cost vs iterations
plt.plot(model.cost_history)
plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function over Iterations")
plt.show()