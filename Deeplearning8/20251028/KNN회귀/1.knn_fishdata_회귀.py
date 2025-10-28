import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from dataset import perch_length, perch_weight

# 모델 입력 데이터(특성 데이터): 농어 길이(perch_length)
# 타깃 데이터: 농어 무게(perch_weight)

# train / test split
train_input, test_input, train_target, test_target = \
       train_test_split(perch_length, perch_weight, random_state=42)

print(train_input.shape) # 부적합한 1차원 배열로 생성됨

# shape를 2차원으로 변경
train_input = train_input.reshape(-1, 1) # -1: 데이터 갯수만큼 알아서 shape 변경
test_input = test_input.reshape(-1, 1)
print(train_input.shape)

# KNN_model_r = KNeighborsRegressor() # KNN 회귀 모델 준비
KNN_model_r = KNeighborsRegressor(n_neighbors=3) # KNN 회귀 모델 준비
KNN_model_r.fit(train_input, train_target) # 길이에 따른 무게 예측

# 성능예측
print(KNN_model_r.score(test_input, test_target)) # 테스트 데이터 성능 R2 성능지표 사용
# 0.992809406101064
print(KNN_model_r.score(train_input, train_target)) # 훈련 데이터 성능 R2 성능지표 사용
# 0.9698823289099254

# 훈련 성능지표가 테스트 성능지표보다 높아야 좋은 모델이다.
# KNN_model_r = KNeighborsRegressor() 결과가 반대로 나왔으므로 좋은 모델이라 볼 수 없고
# 이걸 과소 적합(underfiting)이라고 부른다.
# KNeighborsRegressor(n_neighbors=3) 처럼 기본 5에서 3으로 줄인 경우
# 과소적합 문제가 완화됐음을 확인할 수 있다.

# 테스트 데이터 예측
print(test_input[0:1])
pred = KNN_model_r.predict(test_input[0:1]) # 주변 5개 평균 예측
print(pred) # [60.]
# print(test_target[0:5])

# 뉴 데이터 예측
print(KNN_model_r.predict([[40]])) # 길이가 40인 농어 무게 예측
print(KNN_model_r.predict([[80]])) # 길이가 80인 농어 무게 예측
print(KNN_model_r.predict([[200]])) # 길이가 200인 농어 무게 예측
print(KNN_model_r.predict([[300]])) # 길이가 300인 농어 무게 예측

# 길이가 80 이상 넘어가면 [1033.33333333]만 나오는데,
# 이는 이웃한 주변 5개가 주어진 데이터의 가장 끝 값만 바라보기 때문이다.(예측 오류 한계)
# 이 문제를 해결하려면 선형회구, 다항회귀 등 다른 회귀 모델을 사용해야 한다.