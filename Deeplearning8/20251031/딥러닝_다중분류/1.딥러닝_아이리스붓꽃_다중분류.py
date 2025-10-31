# p145 ~ p145
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시

import pandas as pd
import numpy as np
# 판다스 출력 옵션 조정
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 1000)
# 머신러닝 모델에서 많이 활용한 데이터셋을 이미 포함하고 있음
from sklearn.datasets import load_iris # 붓꽃 데이터셋 가져오기

iris = load_iris() # iris 데이터셋을 dict 형태로 반환
#print(iris)
print(iris['target_names'])
print(iris['feature_names'])

# 사전의 데이터를 추출해서 pandas의 데이터 프레임 객체로 재구성
iris_df = pd.DataFrame(
    np.column_stack([iris['data'], iris['target']]),
    columns=['sepal_len', 'sepal_wd', 'petal_len', 'petal_wd', 'target'])

print(iris_df.head()) # 머리부터 5개만
print("="*50)

x_traindata = iris_df[['sepal_len', 'sepal_wd', 'petal_len', 'petal_wd']] # 특정 컬럼만 선택 추출
print("=> x_traindata")
print(x_traindata[:5])
y_traintarget = iris_df[['target']]
# print(y_traintarget[:5])
print("=> y_traintarget.values")
print(y_traintarget.values)

# 전체 150개 데이터를 섞으면서 train / test 로 데이터셋 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(x_traindata, y_traintarget, random_state=0)

print("=" * 53)
print(len(train_input), len(test_input)) # 112 38
print(train_input[:5]) # 모델에 투입되는 특성 데이터의 개수 확인 ==> 4개
print(train_target[:5])

# 타깃 데이터를 원핫 인코딩 형태로 변환
from tensorflow.keras.utils import to_categorical
train_target_onehot = to_categorical(train_target)
print(train_target_onehot)

# 실제 사용할 타깃 데이터 ==> train_target_onehot

# 딥러닝 다중 분류 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu')) # 입력층 -> 뉴런 갯수 wx + b = 16 * 4 + 16 = 80
model.add(Dense(8, activation='relu')) # 은닉층
model.add(Dense(3, activation='softmax')) # 출력층 -> 답이 '3지선다'이므로 입력 3개

# model.summary()
# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 -> 원핫리코드를 대상으로 설정
model.fit(train_input, train_target_onehot, epochs=50, batch_size=1)

# test 데이터로 모델 성능 평가
test_target_onehot = to_categorical(test_target)
print("test acc: ", model.evaluate(test_input, test_target_onehot)[1])

# 모델 저장
model.save('irisbestmodel.h5')

# 2.딥러닝_아이리스붓꽃_데이터예측.py 파일 생성해 새로운 모델에 새로운 데이터 투입해 예측까지 완료
