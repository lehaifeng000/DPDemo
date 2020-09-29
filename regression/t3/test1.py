# from sklearn.datasets import load_boston
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = np.loadtxt('housing.data', delimiter=',')
print(data.shape)

train_count = 400
test_count = 100
# col_count = 14

train_x = data[:train_count, :-1]
train_y = data[:train_count, -1]

train_y = train_y.reshape(-1, 1)

model = LinearRegression()
model.fit(train_x, train_y)

score = model.score(train_x, train_y)
print(score)
