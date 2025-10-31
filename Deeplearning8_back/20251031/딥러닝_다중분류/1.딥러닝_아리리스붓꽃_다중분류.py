# 1. 데이터셋 준비하고 전처리 ( numpy, pandas 패키지를 활용 )
import pandas as pd
import numpy as np
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)
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

print(len(train_input)) # 112
print(len(test_input))   # 38
print(train_input[:5])  # 모델에 투입되는 특성 데이터의 개수 ==> 4개
print(train_target[:5])

# 타깃 데이터를 원핫 인코딩 형태로 변환
from tensorflow.keras.utils import to_categorical
train_target_onehot = to_categorical(train_target)
print(train_target_onehot)
# 실제 사용할 타깃 데이터 ==> train_target_onehot
test_target_onehot = to_categorical(test_target)
print(test_target_onehot)


# 딥러닝 다중 분류 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add( Dense(16, input_dim=4, activation='relu'))  # 입력층
model.add( Dense(8, activation='relu') )  # 은닉층
model.add( Dense(3, activation='softmax'))  # 출력층

#model.summary()
# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(train_input, train_target_onehot, epochs=50, batch_size=1)

# test 데이터로 모델 성능을 평가
print('test acc : ', model.evaluate(test_input, test_target_onehot)[1] )
# 모델을 저장
model.save('irisbestmodel.h5')
# 2.딥러닝_아이리스붓꽃_데이터예측.py  파일을 새로 만들어서
#  모델에 새로운 데이터를 투입해서 예측 까지 완료!!








