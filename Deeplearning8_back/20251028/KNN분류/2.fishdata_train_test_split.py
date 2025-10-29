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

fish_data = np.column_stack( [ lengh,weight ] )  # 모델에 투입할 입력(특성) 데이터 준비
#subset = fish_data[:5]
# a = np.arange(1,11).reshape(2,5)
# print(a)
# b = np.arange(11,21).reshape(2,5)
# print(b)
a = np.ones((35,))
print(a)
b = np.zeros((14,))
print(b)
fish_target = np.concatenate((a,b)) # 두 배열을 병합
print(fish_target)  # 정답 데이터

# 전체데이터셋을  train / test 데이터셋으로 분할
from sklearn.model_selection import train_test_split

train_input, test_input, train_traget, test_target = \
    train_test_split(fish_data,fish_target, stratify=fish_target, random_state=42)
# 디폴트는 랜덤으로 전체 데이터셋을 무작위 섞은 후 분할 함
print(len(train_input))
print(len(test_input))

# knn 모델 준비
knnmodel = KNeighborsClassifier()
knnmodel.fit(train_input, train_traget)
print( knnmodel.score(test_input, test_target) ) # 성능 평가

print( knnmodel.predict( [ [25,150] ] ) )  # [0] : 빙어

dist , indexs = knnmodel.kneighbors( [ [25,150] ] ) # 주변 5개의 거리와 데이터의 인덱스를 반환
print(dist)
print(indexs)  # 주변 5개 데이터의 인덱스

# 산점도 시각화
# plt.scatter(train_input[:,0] , train_input[:,1])
# plt.scatter(25, 150, marker='^')
#
# # fancy indexing : 배열을 전달해서 특정 데이터를 선택 추출하는 문법
# plt.scatter(train_input[indexs,0], train_input[indexs,1], marker='D')
# plt.show()

arr = np.arange(1,11).reshape(5,2)
print(arr)
# res = arr[[1,3]]  # fancy indexing
# print(res)
# axis = 0 : 행축,  axis = 1 : 열축
print( arr.mean(axis=0) ) # 넘파이 집계(통계) 함수 제공
print( arr.std(axis=0) )
# 표준점수 정규화 ==> (각 특성데이터 - 평균) / 표준편차



# import pandas as pd
# df = pd.DataFrame(subset, columns=['one','two'])
# print(df)
# df.to_csv("D:\\파이썬_강의자료_2025\\KDT_8기\\Deeplearning8\\20251028\\fishdata.csv")

# arr = np.array( [ [1,2,3],[5,6,7] ] )
# print(arr)
# print(arr.ndim ,  arr.shape )
#
# arr = arr.reshape(3,2)
# print(arr)
#
#
# arr1 = np.arange(1, 46).reshape(3,3,5)
# print(arr1)
#
# result = np.array([5,6,7]) + 5
# print(result)