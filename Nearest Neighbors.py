from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=3)
import pandas as pd
house_data = pd.read_csv("E:\\Python Tutorial\\Teacher (ML)\\house_rental_data.csv")
print(house_data.head())
nn.fit(house_data[['Price']])
house_data[:1]['Price']
NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
         metric_params=None, n_jobs=1, n_neighbors=3, p=2, radius=1.0)
print(nn.kneighbors(house_data[:1][['Price']]))
print(house_data.iloc[[354]])

# .iloc → stands for integer-location based indexing
# It allows you to select rows (and columns) by their position number, not by label.
# The double brackets [[354]] are important:
# house_data.iloc[354] → returns a Series (a single row as a 1D object)
# house_data.iloc[[354]] → returns a DataFrame (a 2D object)
