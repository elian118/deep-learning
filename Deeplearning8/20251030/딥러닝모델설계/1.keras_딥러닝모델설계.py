import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시
# 125p
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam # 가장 많이 사용되는 옵티마이저

model = Sequential()
model.add(Dense(30, input_dim=4, activation='relu')) # 첫번째 은닉층 생성하고 추가 
# 참고 input_shape 설정인 경우 input_shape=(4, )처럼 차원 수를 넣어준다
model.add(Dense(10, activation='relu')) # 두번째 은닉층 생성하고 추가
model.add(Dense(1, activation='sigmoid')) # 출력층 생성하고 추가 -> 여기선 시그모이드 함수 설정

print('# 모델 설계 내용 출력')
model.summary() # 모델 설계 내용 출력

# dense 당 파라미터 갯수 = (가중치 * input_dim) + 가중치

# 모델을 사용할 수 있는 환경 구축 ==> compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # 적합 loss 설정값은 126p 하단 박스 참고
    metrics=['accracy']) # 적합 metrics 값은 126p 상단 박스 참고


# # 모델 학습
# model.fit()
#
# # 모델 성능 평가
# model.evaluate()
#
# # 모델 예측
# model.predict()