import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor  # knn 회귀 모델

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 모델 입력 데이터(특성데이터) : 농어 길이 ( perch_length)
# 타깃 데이터 : 농어 무게 ( perch_weight )

# train / test split
train_input, test_input, train_target, test_target = \
    train_test_split(perch_length, perch_weight, random_state=42)

# shape 을 2차원으로 변경
train_input = train_input.reshape(-1,1)  # -1 : 데이터의 개수 만큼 알아서 shape을 변경
test_input = test_input.reshape(-1,1)
print(train_input.shape)

KNN_model_r = KNeighborsRegressor(n_neighbors=3) # KNN 회귀 모델 준비
KNN_model_r.fit(train_input, train_target)

print( KNN_model_r.score(test_input, test_target) )  # 테스트 데이터 성능 R2
print( KNN_model_r.score(train_input, train_target )) # train 데이터 성능
# 테스트 데이터 예측
print(test_input[0:1])
pred = KNN_model_r.predict( test_input[0:1] )
print(pred)

# 뉴 데이터를 예측
print( KNN_model_r.predict( [[40]] ))  # 길이가 40인 농어의 무게를 예측
print( KNN_model_r.predict( [[80]] ))
print( KNN_model_r.predict( [[200]] ))
print( KNN_model_r.predict( [[300]] ))


