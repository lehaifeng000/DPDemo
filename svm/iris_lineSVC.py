# 线性SVM iris 二分类
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
# 0,1分类为0，2分类为1
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([
    # 预处理，归一化
    ("scaler", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])

svm_clf.fit(X, y)
res = svm_clf.predict([[5.5, 1.7]])

print(res)
