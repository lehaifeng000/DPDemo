import tensorflow as tf
from tensorflow import keras
import numpy as np
import resnet 
tf.compat.v1.disable_eager_execution()


cifar10=keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train.astype(np.float)/255.0
x_test=x_test.astype(np.float)/255.0
print(x_train.dtype, type(x_train), x_train.shape)



net=resnet.ResNet18()


net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# net.fit(x_train, y_train, epochs=10, batch_size=32)
net.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test,y_test))

# testLoss, testAcc = net.evaluate(x_test, y_test)
# print(testAcc)

