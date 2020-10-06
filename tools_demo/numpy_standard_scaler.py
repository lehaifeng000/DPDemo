# 使用numpy实现标准差

import numpy as np

x_np = np.array([12,346,214,56,321,110])
x_np.shape = (3, 2)
print(x_np)
# 求均值
mean = np.mean(x_np, axis=0)
# 求标准差
std = np.std(x_np, axis=0)

another_trans_data = x_np - mean
another_trans_data2 = another_trans_data / std
print('标准差标准化的矩阵为：{}'.format(another_trans_data2))