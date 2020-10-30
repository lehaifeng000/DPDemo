import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

# gpus= tf.config.experimental.list_physical_devices('GPU')
gpus= tf.config.list_physical_devices('GPU') # tf2.1版本该函数不再是experimental
# print(gpus) # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
tf.config.experimental.set_memory_growth(gpus[0], True) # 其实gpus本身就只有一个元素

cifar10=keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

import random
index = [i for i in range(len(x_train))] 
random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]


# print(x_train.shape,x_test.shape)

net = keras.models.Sequential([
    # 卷积层1
    tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation="relu", input_shape=(32, 32, 3), padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    # 卷积层2
    tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    # 卷积层3
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    # 卷积层4
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),

    tf.keras.layers.Flatten(),

    # 全连接层1
    tf.keras.layers.Dense(120, activation="relu"),

    # 全连接层2
    tf.keras.layers.Dense(84, activation="relu"),

    # 全连接层3
    tf.keras.layers.Dense(10, activation="softmax")

])

net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

net.fit(x_train, y_train, epochs=5, batch_size=32,validation_split=0.1)

testLoss, testAcc = net.evaluate(x_test, y_test)
print(testAcc)

