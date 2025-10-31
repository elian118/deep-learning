import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(train_input, train_target), (test_input, test_target) = fashion_mnist.load_data() # 28 * 28 이미지 샘플

print(train_input.shape) # (60000, 28, 28)
# CNN은 3차원 데이터만 받으므로 여기에 맞게 변경해준다
# 이미지 데이터 스케일링은 색상 최대값인 255로 나눠주면 0~1 범위로 정규화할 수 있다.
train_input = train_input.reshape(-1, 28, 28, 1) / 255.0 # 3차원으로 변경 및 이미지 데이터 스케일 정규화
print(train_input.shape) # (60000, 28, 28, 1)

# 60000 ==> 분할
from sklearn.model_selection import train_test_split
# train / validation 으로 분할

train_scaled, val_scaled, train_target, val_target = \
    train_test_split(train_input, train_target, test_size=0.2, random_state=42)

print(len(train_scaled), len(val_scaled)) # 48000 12000

# CNN 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

# Conv2D ==> 합성곱층
# MaxPool2D ==> 풀링층

model = Sequential()
model.add(Conv2D(
    32, # filters
    kernel_size= (3, 3), # 또는 3
    padding='same',
    activation='relu',
    input_shape=(28, 28, 1)))
model.add(MaxPool2D(2)) # (2, 2)
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) # 출력층
model.summary()

# 모델 컴파일
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', # 원핫 인코딩이 아니어도 정수이면 계산 가능 
    metrics=['accuracy'])

# 학습도중 손실을 체크해 더 이상 손실이 줄지 않으면 모델 저장하고 즉시 학습 종료(조기종료 콜백)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

cb_modelcheck = ModelCheckpoint('mnistbestmodel.h5') # model.save() 대체
early_stopping = EarlyStopping(patience=3, restore_best_weights=True) # 손실 참기 3번 -> 넘기면 이전 최고 가중치값으로 복구

history = model.fit(
    train_scaled,
    train_target,
    epochs=20,
    validation_data=(val_scaled, val_target), # 학습직후 데이터 유효성 검증 실행
    callbacks=[cb_modelcheck, early_stopping], # 콜백 설정
    verbose=1)

print(history.history['loss']) # train 데이터
print(history.history['accuracy'])
print(history.history['val_loss']) # val 데이터
print(history.history['val_accuracy'])

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
# => val loss가 더 낮아지지 않는 시점의 train_loss 값으로 저장 후 조기 훈련 종료
plt.show() # 조기 종료된 근거인 loss / val_loss 추이 확인 가능