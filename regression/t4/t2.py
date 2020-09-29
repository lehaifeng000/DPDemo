import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.linspace(-10, 10, 100, dtype=float)
y = 2 * x * x + (np.random.rand(x.size).reshape(x.shape) - 0.5) * 3

x = x.reshape(-1, 1)
y_train = y.reshape(-1, 1)

xx1 = np.linspace(-15, 15, 100)

poly = PolynomialFeatures(degree=2)
x_quadratic = poly.fit_transform(x)

model = LinearRegression()

model.fit(x_quadratic, y_train)

xx1 = np.linspace(-15, 15, 100)

xx = poly.transform(xx1[:, np.newaxis])
yy = model.predict(xx)

# plt.plot(x, y, '-g')
plt.plot(xx1, yy, '-r')
plt.show()
# xx = np.linspace(-10, 10, 100)
