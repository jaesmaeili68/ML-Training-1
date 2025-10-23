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

#Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtr, ytr)

#Predict
ypred = model.predict(xte)
#Score
from sklearn.metrics import r2_score
r2 = r2_score(yte, ypred)
print(r2)


