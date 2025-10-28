import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt # 시각화 라이브러리(패키지)
# seaborn, plotly

# 학습시킬 데이터셋 준비
from dataset import bream_length, bream_weight, smelt_length, smelt_weight, length, weight

fish_data = np.column_stack([length, weight]) # 모델에 투입할 입력(특성) 데이터 준비
# subset = fish_data[:5]
# 2차원 배열 생성 예시
# a = np.arange(1, 11).reshape(2, 5)
# print(a)
# b = np.arange(11, 21).reshape(2, 5)
# print(b)
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

print(len(train_input), len(test_input))     # 36 13
print(len(train_target), len(test_target))    # 36 13

# knn 모델 준비
knnmodel = KNeighborsClassifier()
knnmodel.fit(train_input, train_target)
print(knnmodel.score(test_input, test_target)) # 성능 평가 # 1.0

print(knnmodel.predict([[25, 150]])) # [0]: 빙어 -> 잘못된 답 도출
dist, indexs = knnmodel.kneighbors([[25, 150]]) # 주변 5개 거리와 데이터 인덱스 반환
print(dist)
print(indexs) # 주변 5개 데이터 인덱스
# [[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]

# # 산점도 시각화
# plt.scatter(train_input[:, 0], train_input[:, 1]) # 길이와 무게로 잘라서 입력
# plt.scatter(25, 150, marker='^')
#
# # fancy indexing: numpy 배열을 전달해 특정 데이터를 선택 추출하는 문법
# plt.scatter(train_input[indexs, 0], train_input[indexs, 1], marker='D')
# plt.show()


arr = np.arange(1, 11).reshape(5, 2)
print(arr)
# # fancy indexing example
# res = arr[[1, 3], 1] # 1행과 3행만 추출하고 두번째 열만 다시 추출
# print(res)
# axis = 0: 행축, axis = 1: 열축
# 넘파이 집계(통계) 함수 제공 
print(arr.mean(axis=1)) # 열축으로 평균 값 계산해 리스트로 반환
print(arr.std(axis=0)) # 행축으로 표준편차 값 계산해 리스트로 반환
# 표준점수 정규화 ==> (각 특성 데이터 - 평균) / 표준편차

# import pandas as pd
#
# df = pd.DataFrame(subset, columns=['one', 'two'])
# print(df)
# df.to_csv("fishdata.csv")

# arr_x = np.column_stack([[1, 2, 3], [5, 6, 7]])
# print(arr_x)
# [[1 5]
#  [2 6]
#  [3 7]]

# arr_y = np.array([[1, 2, 3], [5, 6, 7]]) # np.row_stack([[1, 2, 3], [5, 6, 7]])
# print(arr_y)
# print(arr_y.ndim, arr_y.shape)
# [[1 2 3]
#  [5 6 7]]
# 2 (2, 3)

# arr_y = arr_y.reshape(3, 2)
# print(arr_y)
# print(arr_y.ndim, arr_y.shape)
# [[1 2]
#  [3 5]
#  [6 7]]
# 2 (3, 2)

# # 3차원 데이터 예시 ex) 색상
# arr1 = np.arange(1, 46).reshape(3, 3, 5)
# print(arr1)
# [[[ 1  2  3  4  5]
#   [ 6  7  8  9 10]
#   [11 12 13 14 15]]
#
#  [[16 17 18 19 20]
#   [21 22 23 24 25]
#   [26 27 28 29 30]]
#
#  [[31 32 33 34 35]
#   [36 37 38 39 40]
#   [41 42 43 44 45]]]

# result = np.array([5, 6, 7]) + 5
# print(result) # [10 11 12]