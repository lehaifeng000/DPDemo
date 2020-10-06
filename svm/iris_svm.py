# SVM iris 二分类

import numpy as np
from sklearn import datasets  # 需要先导入,不能直接sklearn.datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = datasets.load_iris()

X = iris["data"][:100]
# 取0，1两类
y = (iris["target"][:100]).astype(np.float64)

svm_clf = Pipeline([
    # 预处理，归一化
    ("scaler", StandardScaler()),

    ("svc", SVC()),
])

svm_clf.fit(X, y)
res = svm_clf.predict([[5.1, 3.6, 1.3, 0.2]])

print(res)
