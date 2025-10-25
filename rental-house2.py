# using Ridge regression while data are normalized


import numpy as np
import pandas as pd

# data fram
df = pd.read_csv("E:\\Python Tutorial\\Teacher (ML)\\house_rental_data.csv")
print(df.head())
print(df.info())

# x-y separation
x = df.drop('Price', axis=1)
y = df['Price']

# convert data to arrays (hp :house price , hf: house features)
hf = np.array(x)
hp = np.array(y)

# Normalize
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
nhf = Scaler.fit_transform(hf)

# train-test
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(nhf , hp, test_size=0.2)

#Comparing different degrees
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
models = []
for i in range(1,6):
    pol = PolynomialFeatures(i)
    
    xtr_pol = pol.fit_transform(xtr)
    xte_pol = pol.transform(xte)

    lr = Ridge(alpha=0.75)
    models.append(lr)
    lr.fit(xtr_pol, ytr)
    score = lr.score(xte_pol,yte)
    print ('Train Data',i, lr.score(xtr_pol, ytr))
    print ('Test Data', i, score)

#Here you used Ridge Regression (L2 regularization) to prevent overfitting — because coefficients tend to blow up in high-degree polynomial models.
# If R² is too high on train and too low on test, you have overfitting (the model only retained the training data).

print(models[0].coef_)
#In fact, you are taking the coefficients of the first linear model (the first-order model in your loop).