import tensorflow as tf
# gpus= tf.config.experimental.list_physical_devices('GPU')
gpus= tf.config.list_physical_devices('GPU') # tf2.1版本该函数不再是experimental
print(gpus) # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
tf.config.experimental.set_memory_growth(gpus[0], True) # 其实gpus本身就只有一个元素
