# VGG 实例

## vgg16网络结构
1. 第一组
    - 卷积层,kernel_size:3，数量64
    - 卷积层,kernel_size:3，数量64
    - 池化层,pool_size:(2,2)，步长2
2. 第二组
    - 卷积层,kernel_size:3，数量128
    - 卷积层,kernel_size:3，数量128
    - 池化层,pool_size:(2,2)，步长2
3. 第三组
    - 卷积层,kernel_size:3，数量256
    - 卷积层,kernel_size:3，数量256
    - 池化层,pool_size:(2,2)，步长2
4. 第四组
    - 卷积层,kernel_size:3，数量512
    - 卷积层,kernel_size:3，数量512
    - 池化层,pool_size:(2,2)，步长2
5. 第五组
    - 卷积层,kernel_size:3，数量512
    - 卷积层,kernel_size:3，数量512
    - 池化层,pool_size:(2,2)，步长2
6. flatten层，二维张量拉直成一维张量
7. 全连接层，4096个神经元，激活函数relu
8. 全连接层，4096个神经元，激活函数relu
9. 全连接层，n个神经元，激活函数softmax

### cifar10使用vgg16严重过拟合