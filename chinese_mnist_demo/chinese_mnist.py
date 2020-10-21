# import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import re

pattern = 'input_\d+_(\d+)_(?P<label>\d+)'
pat=re.compile(pattern)
# s = 'input_1_1_22.jpg'
# res=pat.match(s)
# print(res.group('label'))

file_path='./data/'

file_names=os.listdir(file_path)


def load_data():
    # shape:(15000,64,64,3)
    img_array=np.zeros((15000,64,64),dtype=np.bool)
    label_array=np.zeros((15000,), dtype=np.int8)
    t_img=np.zeros((64,64), dtype=np.float)
    for index,file_name in enumerate(file_names):
        res=pat.match(file_name)
        label=res.group('label')
        label=int(label)-1
        img=plt.imread(file_path+file_name)
        
        t_img=img/255.0
        # print(t_img.shape,img_array.shape)
        img_array[index]=t_img
        label_array[index]=label
    return img_array,label_array
# np.save('chinese_mnist',img_array)