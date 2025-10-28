import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt # 시각화 라이브러리(패키지)
# seaborn, plotly

# 학습시킬 데이터셋 준비
from dataset import bream_length, bream_weight, smelt_length, smelt_weight, length, weight

fish_data = np.column_stack([length, weight]) # 모델에 투입할 입력(특성) 데이터 준비
# 1차원 배열 생성
a = np.ones((35, ))
print(a)
b = np.zeros((14, ))
print(b)
fish_target = np.concatenate((a, b)) # 두 배열 병합
print(fish_target)

# 전체 데이터셋을 train / test 데이터셋으로 분할
from sklearn.model_selection import train_test_split

# 디폴트 랜덤으로 전체 데이터셋을 무작위로 섞은 후 분할
train_input, test_input, train_target, test_target = \
    train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)

print(len(train_input), len(test_input))    # 36 13
print(len(train_target), len(test_target))  # 36 13

# train/test input data를 표준점수 정규화 전처리 수행
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
# 표준점수 정규화 ==> (각 특성 데이터 - 평균) / 표준편차
train_scale = (train_input - mean) / std
print(train_scale)

# knn 모델 준비
knnmodel = KNeighborsClassifier()
# 표준점수 정규화된(스케일링된) 훈련데이터 입력
knnmodel.fit(train_scale, train_target)
print(knnmodel.score(test_input, test_target)) # 성능 평가

# 새로운 예측 데이터로 표준점수 정규화
newdata = ([25, 150] - mean) / std
print(newdata)
print(knnmodel.predict([newdata])) # [1.] -> 도미

dist, indexs = knnmodel.kneighbors([newdata]) # 주변 5개 거리와 데이터 인덱스 반환
print(dist) # [[0.2873737  0.7711188  0.89552179 0.91493515 0.95427626]]
print(indexs) # 주변 5개 데이터 인덱스 # [[21 14 34 32  5]]

# # 산점도 시각화
plt.scatter(train_scale[:, 0], train_scale[:, 1]) # 길이와 무게로 잘라서 입력
plt.scatter(newdata[0], newdata[1], marker='^')

# # fancy indexing: numpy 배열을 전달해 특정 데이터를 선택 추출하는 문법
plt.scatter(train_scale[indexs, 0], train_scale[indexs, 1], marker='D')
plt.show()