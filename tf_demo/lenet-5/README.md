# tensorflow2  lenet-5网络

## lenet-5网络结构
1. 
    - 卷积层,kernel_size:3，数量6
    - 池化层,pool_size:(2,2)，步长2
2. 
    - 卷积层,kernel_size:3，数量6
    - 池化层,pool_size:(2,2)，步长2
3. 
    - 卷积层,kernel_size:3，数量16
    - 池化层,pool_size:(2,2)，步长2
4. 
    - 卷积层,kernel_size:3，数量16
    - 池化层,pool_size:(2,2)，步长2
5. flatten层，二维张量拉直成一维张量
6. 全连接层，120个神经元，激活函数relu
7. 全连接层，84个神经元，激活函数relu
8. 全连接层，10个神经元，激活函数softmax

- 优化器adam，loss交叉熵损失
- net.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## 数据集说明

### mnist
- 手写数字数据集，包含0~9一共10类  
```python
from tensorflow import keras
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
- x_train.shape:(60000,28,28)
- y_train.shape:(60000,)
- x_test.shape:(10000,28,28)
- y_test.shape:(10000,)


### cifar10
- 10个类别，包括飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
```python
from tensorflow import keras
cifar10=keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
- x_train.shape:(50000,32,32,3)
- y_train.shape:(50000,1)
- x_test.shape:(10000,32,32,3)
- y_test.shape:(10000,1)
