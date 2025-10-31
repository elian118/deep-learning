import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  # 랜덤 배열 데이터 생성
import matplotlib.pyplot as plt  # 시각화
import pandas as pd

Housedf = pd.read_csv('BostonHousing.csv')
print(Housedf)
Housedf.info()
print(Housedf.columns)
print(len(Housedf.columns))
# iloc : interger location,  loc : label location
#  수치 인덱싱 ==> stop -1  까지 포함
train_input = Housedf.iloc[ : , :13] # 수치 인덱스를 이용해서 특정 데이터를 선택 , 추출
train_target = Housedf.iloc[:, 13]

# print(train_input[:5])
# print(train_target[:5])
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
    train_test_split(train_input, train_target, test_size=0.3, random_state=42)

# 데이터 스케일 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add( Dense(30, input_dim=13, activation='relu') )
model.add( Dense(6, activation='relu'))
model.add( Dense(1, activation='linear'))

#model.summary()
# 모델 컴파일
model.compile(optimizer='adam', loss = 'mse' , metrics=['mse'])

# 모델 학습
model.fit( train_scaled , train_target, epochs=500, batch_size=4, verbose=1)

model.save('housemodelbest.h5')


