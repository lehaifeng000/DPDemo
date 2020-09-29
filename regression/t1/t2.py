# 多元线性回归

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

x1 = np.arange(-10, 10, 1, dtype=float)
print(x1.shape)
# b=np.random.rand(x1.size).reshape()
y = x1 + (np.random.rand(x1.size).reshape(x1.shape) - 0.5) * 3
x1 = x1.reshape(-1, 1)
y = y.reshape(-1, 1)

model = LinearRegression()
model.fit(x1, y)
score = model.score(x1, y)
print(score)
pre_y = model.predict(x1)
plt.plot(x1, y, 'o')

plt.plot(x1, pre_y, 'r')
# plt.show()
# plt.savefig("rasult.jpg")
