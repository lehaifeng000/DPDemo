import tensorflow as tf
from tensorflow import keras

cifar10=keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


net = keras.models.Sequential([
    # 卷积层1
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu", input_shape=(32, 32, 3), padding="same"),
    # tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(32, 32, 3), padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 卷积层2
    # tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 卷积层3
    # tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

    tf.keras.layers.Flatten(),

    # 全连接层1
    tf.keras.layers.Dense(32*4*4, activation="relu"),

    # tf.keras.layers.Dropout(0.5),

    # 全连接层2
    tf.keras.layers.Dense(32*2*2, activation="relu"),

    # 全连接层3
    tf.keras.layers.Dense(10, activation="softmax")

])

net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

net.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test,y_test))

# testLoss, testAcc = net.evaluate(x_test, y_test)
# print(testAcc)

