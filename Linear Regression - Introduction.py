# Write a Python program to fit a linear regression model, make predictions,and evaluate its performance using regression metrics.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate linear data
x = np.linspace(1, 10, 1000)
y = 2 * x + 5

# Reshape and split
X = x.reshape(-1, 1)
xtr, xte, ytr, yte = train_test_split(
    x.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(xtr, ytr)

# Predict
ypr_tr = model.predict(xte)
ypr_te = model.predict(xte)

# Evaluate with regression metrics
mse = mean_squared_error(yte, ypr_te)
mae = mean_absolute_error(yte, ypr_te)
r2 = r2_score(yte, ypr_te)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Coefficients
print(
    f"Coefficient (m): {model.coef_[0]:.2f}, Intercept (c): {model.intercept_:.2f}")

# Predict a new value
result = model.predict([[5]])
print(f"Prediction for x=5: {result[0]:.2f}")

# Visualize
plt.scatter(xte, yte, color='red', label='Actual')
plt.plot(xte, ypr_te, color='blue', label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
