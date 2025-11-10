#this code gives a regression model to predict diabete and save the model in a specific patch
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
db = load_diabetes()

# convert to data set
df = pd.DataFrame(db.data, columns=db.feature_names)
df['target']=db.target

#print informations
print(df.head())
print(df.info())
print(df.describe())

# Target evaluation
print(db.target_filename)
print(df['target'].value_counts())

# correlation matrix plot
#plot data to know "which features have strong relationships with the target"
import matplotlib.pyplot as plt
import seaborn as sns

    #Histogram - بررسی توزیع ویژگی‌ها
df.hist(bins=20, figsize=(20,20))
plt.title('distribution of diabete features')
plt.show()

    #Heatmap (to see the relationship between each feature and target)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation with Target')
plt.show()

    #seaborn
sns.pairplot(df[['bmi', 'bp', 's5', 'target']], plot_kws={'alpha':0.5})
plt.show()

#preprocessing
from sklearn.model_selection import train_test_split
x = df.drop('target', axis=1)
y = df['target']

# Normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)

#cross validation
  # model
from sklearn.linear_model import LinearRegression
Lr = LinearRegression()

  # اعتبارسنجی 5 بخشی (5-Fold Cross Validation)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(Lr, x_scaled, y, cv=5, scoring='r2')
print(f"mean R2 = {np.mean(scores):.3f} ± {np.std(scores):.3f}")



#Train/test
xtr, xte, ytr, yte = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
Lr.fit(xtr, ytr)
y_pred = Lr.predict(xte)

# Precision
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(yte, y_pred)
mse = mean_squared_error(yte, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yte, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

##ذخیره مدل بعد از آموزش
import pickle
import os

save_path = r"E:\\Python Tutorial\\Teacher (ML)"
with open(os.path.join(save_path, 'diabet_model.pkl'), 'wb') as file:
    pickle.dump(Lr, file)

with open(os.path.join(save_path, 'scaler_diabet.pkl'), 'wb') as file:
    pickle.dump(scaler, file)

print("Saved in:", save_path)


