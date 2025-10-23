import pandas as pd
import numpy as np

#loading data
df = pd.read_csv ("E:\\Python Tutorial\\Teacher (ML)\\house_rental_data.csv")
df.info()
print(df.head)
df.shape

# divide x and y
x = df.drop('Price', axis=1)
y = df['Price']

# convert data to arrays (hp :house price , hf: house features)
hf = np.array(x)
hp = np.array(y)

# Normalize
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
nhf = Scaler.fit_transform(hf)

#Test & Train
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(nhf, hp)

#transform (just x)
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(3)
xtr_pol = pol.fit_transform(xtr)
xte_pol = pol.transform(xte)

#Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtr_pol, ytr)

#Predict
ypred = model.predict(xte_pol)

#Score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2 = r2_score(yte, ypred)
mae = mean_absolute_error(yte, ypred)
mse = mean_squared_error(yte, ypred)
print(f"r2={r2:.4f}",f"mae={mae:.4f}", f"mse={mse:.4f}")

# R² → The closer it is to 1, the better the model performance.
# MSE → Mean Squared Error (the smaller, the better).
# MAE → Mean Absolute Error (the smaller, the better).
