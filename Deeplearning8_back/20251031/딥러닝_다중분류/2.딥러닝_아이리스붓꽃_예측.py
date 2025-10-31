

import tensorflow as tf
from tensorflow.keras.models import load_model  #  저장된 모델 로드

new_model = load_model('irisbestmodel.h5')  # 설계된 모델과 가중치 까지 모두 불러옴
#new_model.summary()

# 1. 데이터셋 준비하고 전처리 ( numpy, pandas 패키지를 활용 )
import pandas as pd
import numpy as np
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)


# 머신러닝 모델에서 많이 활용한 데이터셋을 이미 포함하고 있음
from sklearn.datasets import load_iris  # 붓꽃 데이터 셋

iris = load_iris()  # iris 데이터셋을 dict 형태로 반환
#print(iris)
print(iris['target_names'])
print(iris['feature_names'])

# 사전의 데이터를 추출해서 pandas의 데이터프레임 객체로 구성
iris_df = pd.DataFrame( np.column_stack( [ iris['data'], iris['target'] ]) ,
                        columns=['sepal_len','sepal_wd','petal_len','petal_wd','target'])
print(iris_df.head()) # 머리부터 5개만
print("="*50)
# print(iris_df.tail()) # 끝에서부터 5개만
# print("="*50)
# print(iris_df.sample(5)) # 랜덤 추출

# 4개 특성중 petal_len, petal_wd 만 학습 데이터의 특성으로 추출해서 준비
x_traindata = iris_df[ ['sepal_len','sepal_wd','petal_len','petal_wd'] ]  # 특정 컬럼을 선택 추출
print(x_traindata[:5])
y_traintarget = iris_df['target']
print(y_traintarget.values)
#
# # 전체 150개 데이터를 섞으면서 train / test 로 데이터셋을 분리
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target = \
    train_test_split(x_traindata, y_traintarget, random_state=0)


preddata = test_input[:5].copy()  # .copy()  :  원 데이터에서 5개 자른 데이터를 완벽한 사본객체로 생성

pred = new_model.predict(preddata.values)

class_info = iris['target_names']

# print(pred)
# print( pred[0] )
# print(np.argmax( pred[0] ))  # np.argmax() : 배열 값 중 가장 큰 값의 인덱스를 반환
# print(class_info[ np.argmax( pred[0] ) ])

for item in pred:
    print( class_info[ np.argmax( item ) ] )

print(test_target[:5])