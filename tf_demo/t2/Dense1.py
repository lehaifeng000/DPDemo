import numpy as np
from tensorflow import keras
import tensorflow as tf

class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """构建所需要的参数"""
        # x * w + b. input_shape:[None, a] w:[a,b]output_shape: [None, b]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],
                                             self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        """完成正向计算"""
        return self.activation(x @ self.kernel + self.bias)


customized_softplus = keras.layers.Lambda(lambda x: tf.nn.softplus(x))

model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu'),
    CustomizedDenseLayer(1),
    customized_softplus,
])