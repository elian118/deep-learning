# 131p
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시
# 125p
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam # 가장 많이 사용되는 옵티마이저
import numpy as np # 랜덤 배열 데이터 생성
import matplotlib.pyplot as plt # 시각화
import pandas as pd # csv 파일 불러오기

housedf = pd.read_csv('BostonHousing.csv')
print(housedf)
housedf.info() # 결측치 없음. 전부 숫자 => 가공 없이 바로 사용 가능
print(housedf.columns)
print(len(housedf.columns))

# iloc: integer location, loc: label location
# 수치 인덱싱 ==> stop -1까지 포함
train_input = housedf.iloc[:, :13] # 수치 행렬 인덱스를 이용해 특정 데이터를 선택, 추출 _.iloc(행 인덱스, 열 인덱스) -> 인덱스는 특정한 범위[start:end]로도 전달 가능
train_target = housedf.iloc[:, 13]
print('# 훈련 데이터 분리')
print(train_input[:5])
print(train_target[:5])

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(train_input, train_target, test_size=0.3, random_state=42)

# print(len(train_input), len(test_input))

# 데이터 스케일 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='linear'))

# print('# 모델 정보 확인')
# model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# 모델 학습
model.fit(train_scaled, train_target, epochs=1000, batch_size=4, verbose=1)
model.save('housemodelbest.h5') # 파일이 만들어졌다면 더 이상 실행 안 해도 된다. → 4.keras_주택가격_예측.py에서 재사용