import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # 선형회귀

from dataset import perch_length, perch_weight

# train / test split
train_input, test_input, train_target, test_target = \
       train_test_split(perch_length, perch_weight, random_state=42)

# shape를 2차원으로 변경
train_input = train_input.reshape(-1, 1) # -1: 데이터 갯수만큼 알아서 shape 변경
test_input = test_input.reshape(-1, 1)
print(train_input.shape)

# 선형회귀 모델 준비 및 학습
lr_model = LinearRegression() # 선형회귀 모델 준비
lr_model.fit(train_input, train_target) # 학습
print(lr_model.score(test_input, test_target))

# 예측
print(lr_model.predict([[50]])) # [1241.83860323] -> 1.241kg

# w(가중치: 기울기), b(편항: 절편) 출력
print(lr_model.coef_, lr_model.intercept_)
# [39.01714496] -709.0186449535474
print(lr_model.coef_ * 50 + lr_model.intercept_) # wx + b -> 특성이 하나 뿐이면 1차 방정식
# [1241.83860323] -> lr_model.predict([[50]])과 결과 동일

# 훈련에 필요한 특성을 지정한 갯수 n만큼 n차 방정식이 됨
# 그래서 출력한 가중치 값은 n개만큼 존재할 수 있으므로 [] 안에 위치하도록 의도됨

plt.scatter(train_input, train_target)
plt.plot([15, 50], [lr_model.coef_ * 15 + lr_model.intercept_, lr_model.coef_ * 50 + lr_model.intercept_])
plt.show()

# 직선의 방정식 wx + b는 x가 너무 작으면 음수가 나오는 문제가 있다.
# 그래서 다항회귀는 곡선의 방정식 (wx * wx) + b 로 변형해 이 문제를 해결한다.