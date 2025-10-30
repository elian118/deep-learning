# 136p
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

titanic_df = pd.read_csv("titanic_passengers.csv") # csv 파일 읽어와 Dataframe 객체 생성
print(titanic_df.head())

# 딥러닝에서는 숫자 데이터만 처리 가능하므로 아래 변환 코드는 비활성화
# train_target = titanic_df['Survived']
# 1: 생존, 0: 별세
# titanic_df['Survived'] = titanic_df['Survived'].map({1: 'survived', 0: 'fail'})
# print('=' * 155)
# print(titanic_df.sample(10))

# gender 컬럼 데이터를 수치 데이터로 변환 -> 해당 값이 구하려는 답 x에 해당하기 때문
titanic_df['gender'] = titanic_df['gender'].map({'female': 1, 'male': 0})
print('=' * 137)
print(titanic_df.sample(10))

# Dataframe 객체 정보 확인 ==> info()
print('=' * 137)
titanic_df.info() # 정보 출력

# Age 컬럼에 결측치 데이터 존제 ==> NAN (not a number)
# 결측치 제거가 필요한 상황
# print(titanic_df['Age'].isnull()) # 데이터 없으면 True 있으면 False 반환 ==> 불린배열
# 불린 배열을 이용해 True인 위치만 추출하는 문법 ==> 불린 색인
print(titanic_df.loc[titanic_df['Age'].isnull(), ['Age']])
print(len(titanic_df.loc[titanic_df['Age'].isnull(), ['Age']])) # 177개 결측치 존재 확인

# 교재는 결측치 Age 값을 평균 값으로 채워넣었지만 여기서는 결측치가 있는 데이터를 배제하는 방식으로 실습
# 결측치가 있는 행 제거 ==> dropna()
# how = 'any': 결측치가 하나라도 있으면 해당 행 삭제
# how = 'all': 해당 행에 모든 데이터 각 결측치 일때 삭제
titanic_df.dropna(subset='Age', how = 'any', inplace=True)

print('=' * 137)
titanic_df.info() # 정보 출력

# print(titanic_df['Pclass']) # 1등석과 2등석 정보만 중요한 역할한다고 가정
# Pclass 데이터를 원핫 인코딩(0과 1의 조합으로만 표현 ex. 100, 010, 001)으로 변환해 1등석과 2등석 컬럼 데이터만 사용
# 해당 컬럼 데이터의 수치 데이터를 원핫 인코딩 형태로 변환 ==> get_dummies()
onehot_pclass = pd.get_dummies(titanic_df['Pclass'], prefix='Class')
print('=' * 137)
print(onehot_pclass.sample(10))

# 넘파이 배열의 병합 ==> concatenate()
# pandas의 병합 ==> concat()
titanic_df = pd.concat([titanic_df, onehot_pclass], axis=1)
print('=' * 137)
print(titanic_df.sample(10))

# 데이터셋 준비
titanic_train_input = titanic_df[['gender', 'Age', 'Class_1', 'Class_2']]
titanic_train_target = titanic_df['Survived']

print('=' * 137)
print(titanic_train_input.head())
print(titanic_train_target.head())

print('=' * 137)
print(len(titanic_train_input))
print(len(titanic_train_target))

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(titanic_train_input, titanic_train_target, random_state=42, shuffle=True)

print('=> ' + 'train_test_split 실행 결과 출력')
print(len(train_input), len(test_input))
print(len(train_target), len(test_target))
print(train_input[:30])

# 입력 (특성) 데이터의 스케일 정규화 필요
from sklearn.preprocessing import StandardScaler # 가장 많이 쓰는 표준 스케일러

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input) # 학습방법 변형
# 동일 데이터를 분할해 사용했음으로 train에서 fit한 scale로
# test 데이터는 변환(trasform)만 수행
test_scaled = scaler.transform(test_input)

print('=> ' + '데이터 스케일 정규화 결과 출력')
print(train_scaled[:10])
print(test_scaled[:10])

# 딥러닝 이진분류 모델 설계
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진분류 시 출력층의 활성화 함수 ==> sigmoid
# model.summary()

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, batch_size=16, epochs=800, verbose=1)

# 모델 평가
print("test acc: ", model.evaluate(test_scaled, test_target)[1])

model.save('titanicbestmodel.h5') # 6.keras_타이타닉_이진분류_예측.py 에서 재사용