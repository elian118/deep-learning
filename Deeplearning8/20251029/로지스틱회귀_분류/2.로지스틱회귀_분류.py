import numpy as np
import pandas as pd

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

titanic_df = pd.read_csv("titanic_passengers.csv") # csv 파일 읽어와 Dataframe 객체 생성
print(titanic_df.head())

# train_target = titanic_df['Survived']
# 1: 생존, 0: 별세
titanic_df['Survived'] = titanic_df['Survived'].map({1: 'survived', 1: 'fail'})
print('=' * 155)
print(titanic_df.sample(10))

# gender 컬럼 데이터를 수치 데이터로 변환 -> 해당 값이 구하려는 답 x에 해당하기 때문
titanic_df['gender'] = titanic_df['gender'].map({'female': 1, 'male': 0})
print('=' * 137)
print(titanic_df.sample(10))

# Dataframe 객체 정보 확인 ==> info()
print('=' * 137)
titanic_df.info() # 정보 출력

# Age 컬럼에 결측치 데이터 존제 ==> NAN (not a number)
# 결측치 제거가 필요한 상황
# print(titanic_df['Age'].isnull()) # 데이터 없으면 True 있으면 False 반환 ==> 불린배열
# 불린 배열을 이용해 True인 위치만 추출하는 문법 ==> 불린 색인
print(titanic_df.loc[titanic_df['Age'].isnull(), ['Age']])
print(len(titanic_df.loc[titanic_df['Age'].isnull(), ['Age']])) # 177개 결측치 존재 확인

# 교재는 결측치 Age 값을 평균 값으로 채워넣었지만 여기서는 결측치가 있는 데이터를 배제하는 방식으로 실습
# 결측치가 있는 행 제거 ==> dropna()
# how = 'any': 결측치가 하나라도 있으면 해당 행 삭제
# how = 'all': 해당 행에 모든 데이터 각 결측치 일때 삭제
titanic_df.dropna(subset='Age', how = 'any', inplace=True)

print('=' * 137)
titanic_df.info() # 정보 출력

# print(titanic_df['Pclass']) # 1등석과 2등석 정보만 중요한 역할
# Pclass 데이터를 원핫 인코딩(0과 1의 조합으로만 표현 ex. 100, 010, 001)으로 변환해 1등석과 2등석 컬럼 데이터만 사용
# 해당 컬럼 데이터의 수치 데이터를 원핫 인코딩 형태로 변환 ==> get_dummies()
onehot_pclass = pd.get_dummies(titanic_df['Pclass'], prefix='Class')
print('=' * 137)
print(onehot_pclass.sample(10))

# 넘파이 배열의 병합 ==> concatenate()
# pandas의 병합 ==> concat()
titanic_df = pd.concat([titanic_df, onehot_pclass], axis=1)
print('=' * 137)
print(titanic_df.sample(10))

# 데이터셋 준비
titanic_train_input = titanic_df[['gender', 'Age', 'Class_1', 'Class_2']]
titanic_train_target = titanic_df['Survived']

print('=' * 137)
print(titanic_train_input.head())
print(titanic_train_target.head())

