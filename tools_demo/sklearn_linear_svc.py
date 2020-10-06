# 使用sklearn实现线性支持向量机

import numpy as np
from sklearn.svm import LinearSVC

x_np = np.array([
    [1.1, 5.1],
    [1.2, 5.0],
    [5.1, 1.],
    [5.3, 2]
])
y_np = np.array([0, 0, 1, 1])

svc_model = LinearSVC(loss='hinge')

svc_model.fit(x_np, y_np)

y_pre = svc_model.predict([[1.3, 5.5], [4.8, 0.9]])
print(y_pre)
