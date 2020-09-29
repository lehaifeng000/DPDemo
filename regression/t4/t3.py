import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# import seaborn as sns
# sns.set()

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
# X_test = [[6], [8], [11], [16]]
# y_test = [[8], [12], [15], [18]]

x = np.linspace(-10, 10, 100, dtype=float)
y = 2 * x * x + (np.random.rand(x.size).reshape(x.shape) - 0.5) * 3

X_train = x.reshape(-1, 1)
y_train = y.reshape(-1, 1)

# 简单线性回归
model = LinearRegression()
model.fit(X_train, y_train)
xx = np.linspace(-15, 15, 100)
# yy = model.predict(xx.reshape(xx.shape[0], 1))
# plt.scatter(x=X_train, y=y_train, color='k')
# plt.plot(xx, yy, '-g')

# 多项式回归
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
# X_test_quadratic = quadratic_featurizer.fit_transform(X_test)
model2 = LinearRegression()
model2.fit(X_train_quadratic, y_train)
xx2 = quadratic_featurizer.transform(xx[:, np.newaxis])
yy2 = model2.predict(xx2)
plt.plot(xx, yy2, '-r')
plt.show()
