import numpy as np
import pandas as pd

# data fram
df = pd.read_csv("E:\\Python Tutorial\\Teacher (ML)\\house_rental_data.csv")
print(df.head())
print(df.info())

# x-y separation
hf = df.drop('Price', axis=1)
hp = df['Price']


# train-test
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(hf , hp, test_size=0.2)

#Comparing different degrees
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
models = []
for i in range(1,6):
    pol = PolynomialFeatures(i)
    
    xtr_pol = pol.fit_transform(xtr)
    xte_pol = pol.transform(xte)

    lr = LinearRegression()
    models.append(lr)
    lr.fit(xtr_pol, ytr)
    score = lr.score(xte_pol,yte)
    print ('Train Data',i, lr.score(xtr_pol, ytr))
    print ('Test Data', i, score)

# If R² is too high on train and too low on test, you have overfitting (the model only retained the training data).