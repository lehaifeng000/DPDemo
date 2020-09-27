import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('iris.data', delimiter=',')

epcd1 = data[:50, 0]
epkd1 = data[:50, 1]
# print(epcd.shape)
plt.plot(epcd1, epkd1, 'ob')

epcd2 = data[50:100, 0]
epkd2 = data[50:100, 1]
plt.plot(epcd2, epkd2, '.r')

epcd3 = data[100:150, 0]
epkd3 = data[100:150, 1]
plt.plot(epcd3, epkd3, '^g')

plt.show()
