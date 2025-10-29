import numpy as np  # 수치 연산에 특화된 라이브러리
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt  # 시각화 라이브러리(패키지)
# seaborn , plotly

# 학습시킬 데이터셋 준비
# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

lengh = bream_length + smelt_length
weight  = bream_weight + smelt_weight

fish_data = [  [l,w] for l, w in zip(lengh, weight) ] # 모델에 투입할 특성데이터
# 특성데이터에 맞는 정답 데이터 준비
fish_target = [1]*35 + [0]*14 # 정답
# knn 모델 준비
knncf = KNeighborsClassifier() # KNN 분류 모델 생성
knncf.fit(fish_data, fish_target) # 학습 과정

print(knncf.score(fish_data, fish_target))  # .95 100%

# 예측
print(knncf.predict([[30,600],[10,11]]))  # [ 1 0 ]

# # 특성 데이터를 산점도 시각화로 출력
plt.scatter(bream_length, bream_weight) # 산점도 시각화 내용을 메모리에 랜더링
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.scatter(10,11, marker='^', c='blue')
plt.show()

