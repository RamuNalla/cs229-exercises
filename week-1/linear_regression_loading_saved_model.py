import numpy as np
import joblib

# Step 1: Generate data
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * x + 4 + np.random.normal(0, 1, size=(100, 1))

# Step 4: Load the model
loaded_model = joblib.load("linear_regression_model.joblib")

# Step 5: Make prediction
x_test = np.array([[5]])
y_pred = loaded_model.predict(x_test)
print(f"Prediction at x = 5: {y_pred[0]:.3f}")