import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

import chinese_mnist

x_train,y_train= chinese_mnist.load_data()

x_train=tf.reshape(x_train,(15000,64,64,1))
y_train=tf.reshape(y_train,(15000,1))

net = keras.models.Sequential([
    # 卷积层1
    tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation="relu", input_shape=(64, 64, 1), padding="same"),
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
    tf.keras.layers.Dense(15, activation="softmax")

])

net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
net.summary()
net.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
