import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

x = np.arange(-10, 10, 1, dtype=float)
print(x.shape)
# b=np.random.rand(x.size).reshape()
y = x + (np.random.rand(x.size).reshape(x.shape) - 0.5) * 3
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)
score = model.score(x, y)
print(score)
pre_y = model.predict(x)
plt.plot(x, y, 'o')

plt.plot(x, pre_y, 'r')
# plt.show()
plt.savefig("rasult.jpg")
