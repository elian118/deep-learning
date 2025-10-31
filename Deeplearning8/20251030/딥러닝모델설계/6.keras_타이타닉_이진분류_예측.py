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

titanic_df['gender'] = titanic_df['gender'].map({'female': 1, 'male': 0})
print(titanic_df.loc[titanic_df['Age'].isnull(), ['Age']])
print(len(titanic_df.loc[titanic_df['Age'].isnull(), ['Age']])) # 177개 결측치 존재 확인

titanic_df.dropna(subset='Age', how = 'any', inplace=True)

print('=' * 137)
titanic_df.info() # 정보 출력

# 해당 컬럼 데이터의 수치 데이터를 원핫 인코딩 형태로 변환 ==> get_dummies()
onehot_pclass = pd.get_dummies(titanic_df['Pclass'], prefix='Class')

# pandas의 병합 ==> concat()
titanic_df = pd.concat([titanic_df, onehot_pclass], axis=1)

# 데이터셋 준비
titanic_train_input = titanic_df[['gender', 'Age', 'Class_1', 'Class_2']]
titanic_train_target = titanic_df['Survived']

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(titanic_train_input, titanic_train_target, random_state=42, shuffle=True)

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

from tensorflow.keras.models import load_model

newmodel = load_model('titanicbestmodel.h5')

pred = newmodel.predict(test_scaled)
print('# predict to test_scaled')
print(pred) # 딥러닝의 이진분류 ==> 마지막 출력된 뉴런이 한개이고 sigmoid 함수를 수행하고 반환된 결과(0~1 사잇값 리스트)
print('#' * 50)
print(pred.flatten())

# 결과 10개만 출력
for i in range(10):
    print("실제 생존여부: {}, 예측 생존확률: {:.2f}%, 예측 생존여부: {}"
          .format(test_target.iloc[i],
                  pred.flatten()[i] * 100,
                  'Survived' if pred.flatten()[i] > 0.5 else 'Fail'))

print('#' * 50)
# test_scaled 말고, 새로운 샘플 데이터를 생성해 예측해보기
print('# test_scaled 말고, 새로운 샘플 데이터를 생성해 예측해보기')
sampledf = pd.DataFrame({'gender': [0, 1, 1], 'Age': [45, 30, 62], 'Class_1': [1, 0, 1], 'Class2_2': [0, 1, 0]})
print(sampledf)

sample_scaled = scaler.transform(sampledf)
print('# sample_scaled')
pred_new = newmodel.predict(sample_scaled)
print(pred_new) # 딥러닝의 이진분류 ==> 마지막 출력된 뉴런이 한개이고 sigmoid 함수를 수행하고 반환된 결과(0~1 사잇값 리스트)
print('# predict sample_scaled')
print(pred_new.flatten())

for i in range(len(pred_new.flatten())):
    print("실제 생존여부: {}, 예측 생존확률: {:.2f}%, 예측 생존여부: {}"
          .format(test_target.iloc[i],
                  pred_new.flatten()[i] * 100,
                  'Survived' if pred_new.flatten()[i] > 0.5 else 'Fail'))