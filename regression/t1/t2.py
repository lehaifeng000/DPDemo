import numpy as np

data = np.loadtxt('1.csv', delimiter=",")
print(data, type(data), data.shape)
print(data[:,1])
