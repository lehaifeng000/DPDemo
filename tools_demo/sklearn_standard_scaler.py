# 使用sklearn实现标准差

import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn

x_np = np.array([12, 346, 214, 56, 321, 110])
x_np.shape = (3, 2)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_np)
print("矩阵初值：{}".format(x_np))
print('该矩阵的均值为：{}\n 该矩阵的标准差为：{}'.format(scaler.mean_, np.sqrt(scaler.var_)))
print('标准差标准化的矩阵为：{}'.format(x_train))

