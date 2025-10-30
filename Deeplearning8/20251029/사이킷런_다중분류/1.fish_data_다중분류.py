import numpy as np
import pandas as pd

# 과학적 표기법 억제
np.set_printoptions(suppress=True)

fishdata = pd.read_csv('fish_data.csv') # CSV 파일 내용을 읽어 Dataframe 객체로 생성

print(fishdata)
print('# fishdata 정보 확인')
fishdata.info()
# fishdata.info() 상 Species 정보는 문자열 데이터로, Dtype이 object로 표현됨 => 이 상태로는 다중분류 불가

# 종별 수 세기
print('# 종별 수 세기')
print(fishdata['Species'].value_counts())
print(fishdata['Species'].unique()) # Set으로 변환해 출력

# 7개 물고기 종류를 분류 ==> 다중분류

# Species ==> 타깃 데이터로 분류
# Weight  Length  Diagonal   Height   Width     ==> 입력 데이터로 준비
print('# 모든 컬럼 정보 출력')
print(fishdata.columns)
# fish_input = fishdata[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].values()
fish_input = fishdata[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print('# Species ==> 인풋 데이터로 분류')
print(fish_input)

print('# Species ==> 타깃 데이터로 분류')
fish_target = fishdata['Species'].to_numpy()
print(fish_target)

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
    train_test_split(fish_input, fish_target, random_state=42)

# 데이터셋 스케일 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)

print('# 데이터셋 스케일 정규화')
print(train_scaled[:5])
print(test_scaled[:5])

# 다중분류 모델 준비
from sklearn.linear_model import LogisticRegression

multi_Lrmodel = LogisticRegression(multi_class='multinomial', C=20, max_iter=1000) # 다중 모델은 최소 1000번은 수행해줘야 잘 학습됨

print('# 다중분류 모델 준비')
print('# 학습')

# 학습
multi_Lrmodel.fit(train_scaled, train_target)

# 모델 성능평가
print('# 모델 성능평가')
print("성능 acc: ", multi_Lrmodel.score(test_scaled, test_target))

# 모델 예측
print('# 모델 예측')
print(multi_Lrmodel.predict(test_scaled[:1])) # test의 첫번째 데이터 예측
print('# 모델 예측 값 확인')
print(multi_Lrmodel.classes_)
print(multi_Lrmodel.predict_proba(test_scaled[:1])) # 소프트맥스 함수의 역할은 이 각 클래스별 예측값들의 합이 1에 수렴하도록 반복 교정하는 데 있다. 보는 내용은 교정된 결과임