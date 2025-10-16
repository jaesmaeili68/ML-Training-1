# Write a Python program to fit a linear regression model, make predictions,and evaluate its performance using regression metrics. (y= ax+b)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# generate random data
x = np.linspace(1, 50, 20)
y = np.linspace(1, 100, 20)

# create the graph
# plt.plot(x, y)
# plt.show()

# Reshape and split
X = x.reshape(-1, 1)
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(xtr, ytr)

# Predict
yptr = model.predict(xtr)
ypte = model.predict(xte)

# Evaluate with regression metrics
r2 = r2_score(yte, ypte)
mse = mean_squared_error(yte, ypte)
mae = mean_absolute_error(yte, ypte)

# show results
print(f"r_score: {r2:0.4f}")
print(f"mean_squared_error: {mse:0.4f}")
print(f"mean_absolute_error: {mae:0.2f}")

# parameters (y = ax + b)
a = model.coef_[0]
b = model.intercept_
print(f"a: {a:.4f}")
print(f"b: {b:.4f}")

# Plot actual vs predicted
plt.scatter(xte, yte, color='red', label='Actual')
plt.plot(xte, ypte, color='blue', label='Predicted', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()
