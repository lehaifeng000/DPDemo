import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_img = tf.reshape(x_train, (60000, 28, 28, 1))
test_img = tf.reshape(x_test, (10000, 28, 28, 1))

net = keras.models.Sequential([
    # 卷积层1
    tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation="relu", input_shape=(28, 28, 1), padding="same"),
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
net.summary()
net.fit(train_img, y_train, epochs=5, batch_size=32)

testLoss, testAcc = net.evaluate(test_img, y_test)
print(testAcc)
