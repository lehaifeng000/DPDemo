import tensorflow as tf
from tensorflow import keras
import numpy as np
import resnet 

# gpus= tf.config.experimental.list_physical_devices('GPU')
gpus= tf.config.list_physical_devices('GPU') # tf2.1版本该函数不再是experimental
# print(gpus) # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
tf.config.experimental.set_memory_growth(gpus[0], True) # 其实gpus本身就只有一个元素

# tf.compat.v1.disable_eager_execution()


cifar10=keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train.astype(np.float)/255.0
x_test=x_test.astype(np.float)/255.0
print(x_train.dtype, type(x_train), x_train.shape)



net=resnet.ResNet34()


net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# net.fit(x_train, y_train, epochs=10, batch_size=32)
net.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test,y_test))

# testLoss, testAcc = net.evaluate(x_test, y_test)
# print(testAcc)

