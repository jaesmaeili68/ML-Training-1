import pandas as pd
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.zeros(10)
c = np.full((4, 4), 5)
d = np.full((4*4), 5)
print(a)
print(b)
print(c)
print(d)
e = np.arange(10, 105, 5)
print(e)
f = np.linspace(start=1, stop=2, num=7)
print(f)
g = np.random.randint(5, 10, size=(3, 4))
print(g)
h = np.random.rand(10)
print(h)
i = np.random.random_integers(50, size=(8, 8))
print(i)

print(i[:2])

a1 = pd.Series(data=[1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(a1)
b1 = pd.Series(data=[10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(b1)

df1 = pd.DataFrame({'col1': a1, 'col2': b1})
print(df1)

Rdf2 = pd.read_csv('E:\\Python Tutorial\\canada\\ENB2012_data.csv')
print(Rdf2.head())
print(Rdf2.info())
print(Rdf2.describe())


import matplotlib.pyplot as plt


# x= [1, 2, 3, 4]
# y = np.sin(x)

# fig,ax = plt.subplots()
# plt.plot(x, y, 'bo')
# plt.show()

# x = np.linspace(0,10,1000)
# fig,ax = plt.subplots()
# ax.plot(x,np.cos(x),label='-sin')
# ax.plot(x,np.sin(x), label='-cos')
# leg =ax.legend()
# plt.show()

np.random.seed(10)
data = np.random.randint(2,20,size=(5,5))
print (data)
print (data [:1])