import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시
# 125p
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam # 가장 많이 사용되는 옵티마이저
import numpy as np # 랜덤 배열 데이터 생성
import matplotlib.pyplot as plt # 시각화

# 랜덤 배열 데이터 생성
train_x = np.linspace(0, 10, 10) # 0~10 사이 값으로 채운 10개행 np list
train_y = train_x + np.random.randn(*train_x.shape)
print('# 랜덤 배열 데이터 생성')
print(train_x)
print(train_y)

# plt.scatter(train_x, train_y)
# plt.show()

model = Sequential()
# 편향 미적용 use_bias=False
model.add(Dense(1, input_dim=1, activation='linear', use_bias=True))
model.summary()

model.compile(optimizer='adam', loss='mse')
weight = model.layers[0].get_weights()
print("학습 전 가중치: ", weight[0][0][0])
print("학습 전 편향: ", weight[1][0])


# 학습 => verbose=1 => 학습과정을 로그로 출력
# batch_size, epochs 값을 변경해가며 최적의 가중치(기울기)를 구할 수 있도록 해야 한다.
# model.fit(train_x, train_y, batch_size=2, epochs=500, verbose=1)
model.fit(train_x, train_y, batch_size=4, epochs=10000, verbose=1)

model.compile(optimizer='adam', loss='mse')
weight = model.layers[0].get_weights()
print("학습 후 가중치: ", weight[0][0][0])
print("학습 후 편향: ", weight[1][0])
# 시각화용 변수로 재선언
w = weight[0][0][0]
bias = weight[1][0]

plt.plot(train_x, train_y, label='data')
plt.plot(train_x, train_x * w + bias, label='predict')
plt.legend() # 라벨 표시
plt.show()