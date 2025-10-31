import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  # 랜덤 배열 데이터 생성
import matplotlib.pyplot as plt  # 시각화

train_x = np.linspace(0,10,10)
print(train_x) # 10개

train_y = train_x + np.random.randn(*train_x.shape)
print(train_y)

model = Sequential()
model.add( Dense(1, input_dim=1, activation='linear', use_bias=True))

#model.summary()
model.compile( optimizer='adam', loss = 'mse')

weight = model.layers[0].get_weights()
print("학습 전 가중치 : ", weight[0][0][0])

model.fit(train_x, train_y, batch_size=2, epochs=500, verbose=1)

weight = model.layers[0].get_weights()
print("학습 후 가중치 : ", weight[0][0][0])
w =  weight[0][0][0]
bias =  weight[1][0]
print('bias : ', bias)

plt.plot(train_x, train_y, label='data')
plt.plot(train_x, train_x*w + bias, label='predict')
plt.legend() # 라벨 표시
plt.show()
