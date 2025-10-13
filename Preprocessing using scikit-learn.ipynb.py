from sklearn import preprocessing
import numpy as np

X_train = np.array([[1., -1.,  2.],
                   [2.,  0.,  0.],
                   [0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)
# The line of code X_scaled = preprocessing.scale(X_train) is used to perform standardization on your training data.
# Specifically, it implements Z-score normalization, transforming the data so that it has a mean of 0 and a standard deviation of 1.
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
