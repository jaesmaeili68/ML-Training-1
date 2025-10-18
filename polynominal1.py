# polynomial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# data
x = np.linspace(1, 100, 20)
X = x.reshape(-1, 1)
y = 2*x*x + 6*x - 3

# train-test devision
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

# polynomial feature transformation تبدیل چندجمله‌ای (درجه 2)
pol = PolynomialFeatures(degree=2)
# note: when we say  PolynomialFeatures(degree=2), python knows that the equation is y = ax^2 + bx^1 + cx^0
# So, the generated data will be [x^0   X^1   X^2]  ---->   [1   X^1   X^2]
X_tr_poly = pol.fit_transform(x_tr)
x_te_poly = pol.transform(x_te)

# model
model = LinearRegression()
model.fit(X_tr_poly, y_tr)

# prediction on data
X_plot = np.linspace(1, 100, 200).reshape(-1, 1)   # برای نمودار صاف
X_plot_poly = pol.transform(X_plot)
y_plot = model.predict(X_plot_poly)

# plot
plt.scatter(X, y, color='blue', label='real data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='predicted data')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression (degree=2)")
plt.legend()
plt.grid(True)
plt.show()


# | Step | Tool               | What it does                                              |
# | ---- | ------------------ | --------------------------------------------------------- |
# | 1️⃣  | PolynomialFeatures | Creates new columns like: 1, x, x², x³, ...               |
# | 2️⃣  | LinearRegression   | Learns the best coefficients (a₀, a₁, a₂, …) to predict y |
