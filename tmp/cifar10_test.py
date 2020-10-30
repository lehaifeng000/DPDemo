import tensorflow as tf
from tensorflow import keras
from collections import Counter
cifar10=keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)
y_train=y_train.tolist()
s=list()
# print(y_test)
for i in y_train:
    s.append(i[0])
res=Counter(s)
print(dict(res))


