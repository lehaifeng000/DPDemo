import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=',')
freq = data[:,2]
shengya = data[:,-1]

print(freq)
print(shengya)

plt.plot(freq,shengya,'o')
plt.show()

