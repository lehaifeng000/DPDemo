# SVM iris 多分类

import numpy as np
from sklearn import datasets  # 需要先导入,不能直接sklearn.datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = datasets.load_iris()

x = []
y = []

x.append(iris["data"][:50])
x.append(iris["data"][50:100])
x.append(iris["data"][100:])

y.append(iris['target'][:50])
y.append(iris['target'][50:100])
y.append(iris['target'][100:])

# 每两个类别产生一个分类器
models = []
for i in range(3):
    for j in range(i + 1, 3):
        X = np.vstack((x[i], x[j]))
        Y = np.hstack((y[i], y[j]))
        svm_model = Pipeline([
            # 预处理，归一化
            ("scaler", StandardScaler()),
            # Support Vector Classification
            ("svc", SVC()),
        ])
        svm_model.fit(X, Y)
        models.append(svm_model)

x_test = np.array([[5.1, 3.6, 1.3, 0.2]])

# 使用所有分类器进行分类，投票
result = []
for model in models:
    res = model.predict(x_test)
    result.append(res[0])

from collections import Counter

t = Counter(result)
print(t)
print(t.most_common(1)[0][0])
