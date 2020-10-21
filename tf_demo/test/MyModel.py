import tensorflow as tf
from tensorflow import keras

from MyDense import MyDense


class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = MyDense(512)
        self.fc2 = MyDense(256)
        self.fc3 = MyDense(128)
        self.fc4 = MyDense(10)

    def call(self, input):
        self.fc1.out = self.fc1(input)
        self.fc2.out = self.fc2(self.fc1.out)
        self.fc3.out = self.fc2(self.fc2.out)
        self.fc4.out = self.fc2(self.fc3.out)
        return self.fc4.out


myModel = MyModel()
myModel.build(input_shape=(None, 784))
myModel.summary()
