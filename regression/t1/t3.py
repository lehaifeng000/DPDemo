import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x1 = np.linspace(-10, 10, 100, dtype=float)
x2 = np.linspace(1, 5, 100, dtype=float)
# print(x1.shape)
# b=np.random.rand(x1.size).reshape()
y = 2 * x1 + 0.3 * x2 + (np.random.rand(x1.size).reshape(x1.shape) - 0.5) * 3
x1 = x1.reshape(-1, 1)
x2 = x2.reshape(-1, 1)
# 合并
x = np.hstack((x1, x2))
print(x.shape)
y = y.reshape(-1, 1)

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

xx = np.linspace(-10, 10, 100)
