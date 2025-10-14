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

# Create a transformer object
transformer = preprocessing.StandardScaler()
print(transformer.fit(X_train))
# DO the data transformation
print(transformer.transform(X_train))
# Combination of above twp
print(transformer.fit_transform(X_train))


# Converts all data between 0 & 1 range ( inclusive)
# a type of normalization that scales your data into the range [−1,1]
maxabsscaler = preprocessing.MaxAbsScaler()
print(maxabsscaler.fit_transform(X_train))


# Non-linear transformation
transformer = preprocessing.StandardScaler()
norm = preprocessing.Normalizer()
print(norm.fit_transform(X_train))

# Binarization: Given any data, converts it into 0 & 1
binarizer = preprocessing.Binarizer()
print(binarizer.fit_transform(X_train))
