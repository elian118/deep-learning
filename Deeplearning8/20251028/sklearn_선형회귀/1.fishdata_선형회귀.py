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

# 교재 80page - 다항회귀
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
print(train_poly.shape)

# 선형회귀 모델 준비 및 학습(lr(학습률: learning rate))
lr_model = LinearRegression() # 선형회귀 모델 준비 --> 다항회귀(제곱특성 추가해서)
# lr_model.fit(train_input, train_target) # 학습
lr_model.fit(train_poly, train_target) # 학습
print(lr_model.score(test_poly, test_target))

# 예측
# print(lr_model.predict([[50]])) # [1241.83860323] -> 1.241kg
print(lr_model.predict([[60 ** 2, 50]])) # [1573.98423528] -> 1.573kg

# w(가중치: 기울기), b(편항: 절편) 출력
print(lr_model.coef_, lr_model.intercept_)
# [39.01714496] -709.0186449535474
# 다항회귀 -> [  1.01433211 -21.55792498] 116.05021078278259
print(lr_model.coef_ * 50 + lr_model.intercept_) # wx + b -> 특성이 하나 뿐이면 1차 방정식
# [1241.83860323] -> lr_model.predict([[50]])과 결과 동일
# 다항회귀 -> [ 166.76681625 -961.84603816]
print(lr_model.coef_[0] * 50 ** 2 + lr_model.coef_[1] * 50 + lr_model.intercept_) # wx**2 + wx + b
# 다항회귀 -> 1573.9842352827407

# 다항회귀 시각화
plt.scatter(train_input, train_target)
xpoint = np.arange(15, 50)
plt.plot(xpoint, 1.01 * xpoint ** 2 - 21.6 * xpoint + 116.06)
plt.scatter(60, 2474, marker='^')
plt.show()

# 훈련에 필요한 특성을 지정한 갯수 n만큼 n차 방정식이 됨
# 그래서 출력한 가중치 값은 n개만큼 존재할 수 있으므로 [] 안에 위치하도록 의도됨

# 선형회귀 시각화
# plt.scatter(train_input, train_target)
# plt.plot([15, 50], [lr_model.coef_ * 15 + lr_model.intercept_, lr_model.coef_ * 50 + lr_model.intercept_])
# plt.show()

# 직선의 방정식 wx + b는 x가 너무 작으면 음수가 나올 수 있는 문제가 있다.
# 수학적으로 문제 없으나 현실의 데이터 상으로는 있을 수 없는 값임.
# 그래서 다항회귀는 곡선의 방정식 (wx * wx) + b 로 변형해 기울기(w)가 가장 0에 가까운 것을 찾아내 문제를 해결한다.
# 곡선의 방정식에서 경사 하강법으로 찾아낸다. MSE 공식이 적용되며 이건 곡선그래프에서 순간 기울기를 계산하는 미분 공식이다.
# 기울기(loss)가 최저 지점인 곳을 찾을때가지 계산 반복하다 더 이상 변화가 없는 지점이 나오면 0에 가까운 최저지점으로 판단해 조기종료 시킴
# 자세한 내용은 MSE_경사하강_예제.py 확인