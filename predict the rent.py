# using Ridge regression while data are normalized


from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# data fram
df = pd.read_csv("E:\\Python Tutorial\\Teacher (ML)\\house_rental_data.csv")
print(df.head())
print(df.info())

# x-y separation
hf = df.drop('Price', axis=1)
hp = df['Price']

# # convert data to arrays (hp :house price , hf: house features)
# hf = np.array(x)
# hp = np.array(y)

# Normalize
Scaler = StandardScaler()
nhf = Scaler.fit_transform(hf)

# train-test
xtr, xte, ytr, yte = train_test_split(nhf, hp, test_size=0.2)


pol = PolynomialFeatures(2)
xtr_pol = pol.fit_transform(xtr)
xte_pol = pol.transform(xte)
lr = Ridge(alpha=0.75)
lr.fit(xtr_pol, ytr)
score = lr.score(xte_pol, yte)
print('Train Data', lr.score(xtr_pol, ytr))
print('Test Data', score)

# Here you used Ridge Regression (L2 regularization) to prevent overfitting — because coefficients tend to blow up in high-degree polynomial models.
# If R² is too high on train and too low on test, you have overfitting (the model only retained the training data).


# try to predict a price
# === USER INPUT LOOP ===
print("\nEnter the following house details:")
# Creates an empty dictionary to store the user’s input values. #Each key is a column name (feature), and the value is a list (with one element) so it matches the DataFrame structure later.
user_data = {}
# This loop goes through every column name in your dataset’s features (hf.columns).
for col in hf.columns:
    # float() → converts the input string into a numerical value (since model features are numeric).
    val = float(input(f"{col}: "))
    # stores that number inside a list so we can easily convert it into a one-row DataFrame.
    user_data[col] = [val]

# Convert to DataFrame #Converts the dictionary into a DataFrame, so its structure (columns) matches your training dataset exactly.
new_house = pd.DataFrame(user_data)

# === Scale and Transform ===
# Applies the same normalization that was used during training.
new_house_scaled = Scaler.transform(new_house)
# Expands the scaled features into polynomial features of the same degree used during training.
new_house_pol = pol.transform(new_house_scaled)

# === Predict ===
predicted_price = lr.predict(new_house_pol)
print(f"\n🏠 Predicted house price: ${predicted_price[0]:,.2f}")
