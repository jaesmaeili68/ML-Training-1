import pandas as pd

# data
df = pd.read_csv('E:\\Python Tutorial\\Teacher (ML)\\HR_comma_sep.csv')
# print(df.head())
# print(df.keys())

print(df.sales.unique())
# .unique() → is a Pandas Series method that returns an array of all unique values in that column (no duplicates).
print(df.salary.unique())

# LableEncoder: for converting categorical data into numerical form in machine learning.(converts each unique string label into an integer.)
from sklearn.preprocessing import LabelEncoder
L_enc = LabelEncoder()
df['sales'] = L_enc.fit_transform(df.sales)
df['salary'] = L_enc.fit_transform(df.salary)


f = df.corr()['left'] 
#tells you which features are most strongly associated with employee turnover, and in which direction.
#This computes the correlation matrix for all numerical columns in your dataset.
print(f)

import numpy as np
imp_features = np.abs(f).sort_values(ascending=False).index.tolist()[1:5] #it’s selecting the most important features correlated with the target (left).
#np.abs(f) -> Takes the absolute value of each correlation.
#.sort_values(ascending=False) -> Sorts features by strength of correlation (from highest to lowest).
#.index.tolist() -> Extracts the column names (feature names) as a Python list.
#[1:5] -> Then selects the next 4 most correlated features.

print(imp_features)

from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(df[imp_features] , df['left'], test_size=0.2)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xtr, ytr)

print(lr.score(xte, yte))
print(xtr[0:5])

# predict
print(lr.predict([[0.5, 0, 4, 200]]))

# left value	Meaning
# 0	The employee stayed (did not leave the company)
# 1	The employee left the company