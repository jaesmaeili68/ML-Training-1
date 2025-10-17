import numpy as np

# technique 1
# x = np.linspace(start, stop, num), num means the number of generated numbers between start and stop
# great for smooth curve
x = np.linspace(1, 100, 200)
print(x)

# technique 2
# x = np.arange(start, stop, step), increases by step from start to stop (stop point excluded)
# great when you know the step size
y = np.arange(1, 100, 1)
print(y)

# technique 3
# x = np.random.randit(low, high, size), generates random integers
# great for simulating discrete data
z = np.random.randint(1, 100, 10)
print(z)

# technique 4
# x = np.random.normal(mean, std, size), normally distributed numbers
# generates a NumPy array of 10 random floating-point numbers drawn from a Normal (Gaussian) distribution.
# mean: Can be any float or integer. Does not have to be 0.
# great for testing models or simulating moise
d = np.random.normal(0, 100, 10)
print(d)

# technique 5
# x = np.random.unifrom(low, high, size), random floating-point numbers
# mean: Can be any float or integer. Does not have to be 0.
# great for creating synthetic continous data
e = np.random.uniform(1, 10, 1000)
print(e)


# technique 6
# x = range(low, high), create integers low through high
# mean: Can be any float or integer. Does not have to be 0.
# great for creating synthetic continous data
# When you create f = range(1, 10), you are creating a Python range object, not a list of numbers.
# The range object is a memory-efficient sequence generator. It stores only the start, stop, and step values (which are 1, 10, and 1, respectively, in this case).
# When you print the object itself, Python shows you its description: range(start, stop).
# NOTE: The numbers are generated only when you iterate over the object (e.g., in a for loop) or explicitly convert it to a list.

c = range(1, 10)
print(c)

print(x.shape)
print(y.shape)
print(z.shape)
print(d.shape)
print(e.shape)

print(type(x))
print(type(y))
print(type(z))
print(type(d))
print(type(e))
print(type(c))

