import numpy as np
import matplotlib.pyplot as plt
# import sklearn
from sklearn.linear_model import LinearRegression

data = np.loadtxt('airfoil_self_noise.dat', delimiter=',')

train_count = 1000
total_count = 1100
train_x = data[:train_count, :-1]
train_y = data[:train_count, -1]

test_x = data[train_count:total_count, :-1]
test_y = data[train_count:total_count, -1]

train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

ir_model = LinearRegression()
ir_model.fit(train_x, train_y)

score = ir_model.score(train_x, train_y)
print(score)

test_score = ir_model.score(test_x, test_y)
print(type(ir_model))
print(test_score)
# pre_y = ir_model.predict(test_x)
#
# plt.plot(range(100), test_y, 'o')
# plt.plot(range(100), pre_y, '^')
# plt.show()
