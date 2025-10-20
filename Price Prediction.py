# House Price Prediction
import pandas as pd
import numpy as np

house_data = pd.read_csv(
    'E:\\Python Tutorial\\Teacher (ML)\\house_rental_data.csv', index_col='Unnamed: 0')
print(house_data.head())
house_data.info()
print(house_data.columns)

# separation x= house_features , y= price
house_features = house_data.drop('Price', axis=1)
price = house_data['Price']

# feature_data = house_data[['Sqft', 'Floor', 'TotalFloor', 'Bedroom', 'Living.Room', 'Bathroom',]]
# target_data = house_data['Price']
# print(feature_data)
# print(target_data)

# convert data to arrays
x = np.array(house_features)
y = np.array(price)
# print(x)
# print(y)

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)


# train & test
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(X,y, test_size=0.2)

#transform (just x)
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(3)
xtr_pol = pol.fit_transform(xtr)
xte_pol = pol.transform(xte)


#Model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtr_pol, ytr)



#Predict
yprtr = model.predict(xtr_pol)
yprte = model.predict(xte_pol)


#Accuracy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2 = r2_score(yte, yprte)
mse = mean_squared_error(yte, yprte)
mae = mean_absolute_error(yte, yprte)
print (f"r2={r2:0.4f}, mse={mse:0.4f}, mae={mae:0.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(yte, yprte, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Polynomial Regression - House Price Prediction")
plt.legend()
plt.show()