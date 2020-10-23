import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

cifar10=keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

net.fit(x_train, y_train, epochs=5, batch_size=32)

testLoss, testAcc = net.evaluate(x_test, y_test)
print(testAcc)
