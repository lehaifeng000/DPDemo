import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

cifar10=keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape,x_test.shape)



net = keras.models.Sequential([
    # 第一组
    # 卷积层
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", input_shape=(32, 32, 3), padding="same"),
    # 卷积层
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
    # 池化层
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    # 第二组
    # 卷积层
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"),
    # 卷积层
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"),
    # 池化层
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    # # 第三组
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"),
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"),
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same"),
    # # 池化层
    # tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    # # 第四组
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
    # # 池化层
    # tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    # # 第五组
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
    # # 卷积层
    # tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu", padding="same"),
    # # 池化层
    # tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    # flatten层
    tf.keras.layers.Flatten(),

    # 全连接层1
    tf.keras.layers.Dense(4096, activation="relu"),
    # 全连接层
    tf.keras.layers.Dense(4096, activation="relu"),
    # 全连接层3
    tf.keras.layers.Dense(10, activation="softmax")

])

net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

net.fit(x_train, y_train, epochs=20, batch_size=32)

testLoss, testAcc = net.evaluate(x_test, y_test)
print(testAcc)

