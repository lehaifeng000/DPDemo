# https://blog.csdn.net/qq_36327687/article/details/84943321

import pandas as pd
import numpy as np

data = pd.read_csv('1.csv', header=None)
print(type(data))
print(data.shape)
print(data)
print()
