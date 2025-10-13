from sklearn.feature_selection import VarianceThreshold
import numpy as np
x = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0],
             [1, 1, 1], [0, 1, 0], [0, 1, 1]])
print(x[:, 0])
y = np.var(x[:, 0])
print(y)
z = y*x
# print(z)
sel = VarianceThreshold(0.2)
print(sel)
