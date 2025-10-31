import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


model = Sequential()
model.add( Dense(30, input_dim=4,  activation='relu') )  # 첫번째 은닉층 생성하고 추가
model.add( Dense(10, activation='relu') ) # 두번째 은닉층 생성하고 추가
model.add( Dense(1, activation='sigmoid'))

#model.summary() # 모델 설계 내용을 출력

# 모델을 사용할 수 있는 환경을 구축 ==> compile
model.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'] )  # loss function, optimizer 설정
#
# # 모델 학습
# model.fit()
#
# # 모델 성능 평가
# model.evaluate()
#
# # 모델 예측
# model.predict()