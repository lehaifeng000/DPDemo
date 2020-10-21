import tensorflow as tf
from tensorflow import keras


class MyDense(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_variable(
            name='w',
            shape=[input_shape[-1, self.units]],
            initializer=tf.initializers.RandomNormal()
        )
        self.b = self.add_variable(
            name='b',
            shape=[self.units],
            initializer=tf.initializers.Zeros()
        )

    def call(self, input):
        return input @ self.w + self.b
        return tf.nn.relu(input @ self.w + self.b)
