import numpy as np
import matplotlib.pyplot as plt
# import sklearn
from sklearn.linear_model import LinearRegression

data = np.loadtxt('iris.data', delimiter=',')

epcd1 = data[:50, 0]
epkd1 = data[:50, 1]

train_x = epcd1[:40]
train_y = epkd1[:40]
test_x = epcd1[40:]
test_y = epkd1[40:]

# train_x

train_x = train_x.reshape(-1, 1)
train_y = train_y.reshape(-1, 1)
test_x = test_x.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
irmodel = LinearRegression()

irmodel.fit(train_x, train_y)
res = irmodel.score(train_x, train_y)
print(res)
pre_y = irmodel.predict(test_x)

print("系数")
w = irmodel.coef_
b = irmodel.intercept_
print(w, b)

plt.plot(test_x, test_y, 'o')
plt.plot(test_x, pre_y, '^')
plt.plot(test_x, w*test_x+b)
plt.show()
