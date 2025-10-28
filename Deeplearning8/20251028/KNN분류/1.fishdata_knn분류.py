import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt # 시각화 라이브러리(패키지)
# seaborn, plotly

# 학습시킬 데이터셋 준비
from dataset import bream_length, bream_weight, smelt_length, smelt_weight, length, weight

print(length[:5], len(length))
print(weight[:5], len(weight))

# 모델에 투입할 특성 데이터
fish_data = [[l, w] for l, w in zip(length, weight)] # zip → 속성으로 모아 튜플로 생성 → list 생성 및 구조분해 할당
# 특성 데이터에 맞는 정답 데이터 준비
fish_target = [1] * 35 + [0] * 14 # 정답 선언: [도미1, ..., 도미35, 빙어1, ..., 빙어14]

print(fish_data)
print(fish_target)

# knn 모델 준비
knncf = KNeighborsClassifier() # KNN 분류 모델(객체) 생성
knncf.fit(fish_data, fish_target) # 학습 과정

print(knncf.score(fish_data, fish_target)) # 1.0

# 예측
print(knncf.predict([[30, 600], [10, 11]])) # [1 0] -> [도미, 빙어]

# 특성 데이터를 산점도 시각화로 출력
# 산점도 시각화 내용을 메모리에 랜더링
plt.scatter(bream_length, bream_weight) # 도미 데이터셋 추가
plt.scatter(smelt_length, smelt_weight) # 빙어 데이터셋 추가

# 예측 데이터셋 추가
plt.scatter(30, 600, marker='^')
plt.scatter(10, 11, marker='^', c='blue')

plt.show() # GUI로 출력