import numpy as np
import pandas as pd

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

titanic_df = pd.read_csv("titanic_passengers.csv") # csv파일 읽어와서 Dataframe 객체 생성
print(titanic_df.head())

#train_target = titanic_df['Survived']
# 1 : 생존
# 0 : 별세
titanic_df['Survived'] = titanic_df['Survived'].map( {1:'survival', 0:'fail'} )

# gender 컬럼 데이터를 수치 데이터로 변환
titanic_df['gender'] = titanic_df['gender'].map({'female':1, 'male':0})
print(titanic_df.sample(10))

# Dataframe 객체의 정보를 확인 ==> info()
titanic_df.info()

# Age 컬럼에 결측치 데이터가 존재 ==> NAN ( not a number )
# 결측치 제거가 필요한 상황
# 결측치 검사
#print( titanic_df['Age'].isnull() ) # 데이터가 없으면 True, 데이터가 있으면 False 반환 ==> 불린배열
# 불린배열을 이용해서 True인 위치만 추출하는 문법 ==> 불린 색인
#print ( len( titanic_df.loc[ titanic_df['Age'].isnull() , ['Age'] ] ) ) # 177개의 결측치 존재

# 결측치가 있는 행을 제거  ==> dropna()
# how = 'any' : 결측치가 하나로도 있으면 해당 행 삭제,
# how = 'all' : 해당 행에 모든 데이터 각 결측치 일때 삭제
titanic_df.dropna(subset='Age', how = 'any', inplace=True)
titanic_df.info()

#print(titanic_df['Pclass'])  # 1등석과 2등석 정보만 중요한 역할
#  Pclass 데이터를 원핫 인코딩으로 변환해서 1등석과 2등석 컬럼 데이터만 사용
# 해당 컬럼 데이터의 수치 데이터를 원핫 인코딩 형태로 변환 ==> get_dummies()
onehot_pclass = pd.get_dummies(titanic_df['Pclass'], prefix='Class')
print(onehot_pclass)

# 넘파이 배열의 병합 ==> concatenate()
# pandas의 병합 ==> concat()
titanic_df = pd.concat( [titanic_df, onehot_pclass] , axis= 1)
print(titanic_df)

# 데이터셋 준비
titanic_train_input = titanic_df[ ['gender','Age','Class_1','Class_2'] ]
titanic_train_target = titanic_df['Survived']

print(titanic_train_input.head())

