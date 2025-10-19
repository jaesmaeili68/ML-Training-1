# polynomial
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# data
x = np.random.randint(1, 100, 50)
X = x.reshape(-1, 1)
y = 0.5*x*x + 5*x - 1

# train and devision
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.5)

# polynimial trnsformation
pol = PolynomialFeatures(degree=2)
xtr_pol = pol.fit_transform(xtr)
xte_pol = pol.transform(xte)
# Note
# ✅ fit_transform() → learns the transformation rules and applies them
# ✅ transform() → applies the same rules (already learned) to new data
# fit() → tells the transformer how to expand or scale your data based on training data.
# transform() → actually performs that transformation.


# model
model = LinearRegression()
model.fit(xtr_pol, ytr)


# predict
x_seq = np.linspace(1, 100, 50).reshape(-1, 1)
# x_seq = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
x_seq_poly = pol.transform(x_seq)
y_seq = model.predict(x_seq_poly)


# statistics # metric(y_true, y_pred)
y_pred = model.predict(xte_pol)
r2 = r2_score(yte, y_pred)
mse = mean_squared_error(yte, y_pred)
mae = mean_absolute_error(yte, y_pred)
print(f"r2 = {r2:0.2f}, mse = {mse:0.2f}, mae = {mae:0.2f}")

# coefficients
print("Intercept (b₀):", model.intercept_)
print("Coefficients:", model.coef_)

#
# Equation
# Get the feature names for each polynomial term
feature_names = pol.get_feature_names_out(['x'])

# Combine coefficients with names
coefs = model.coef_
intercept = model.intercept_
print(f"Model Equation: y = {intercept:.2f} ", end='')
for i in range(1, len(coefs)):
    power = i
    coef = coefs[i]
    sign = '+' if coef >= 0 else '-'
    print(f"{sign} {abs(coef):.2f}x^{power} ", end='')
#

# plot
plt.scatter(x, y, color='red', label='Data')
plt.plot(x_seq, y_seq, color='blue', label='Polynomial fit')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression (degree=2)")
plt.show()
