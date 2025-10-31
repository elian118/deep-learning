import numpy as np
import pandas as pd

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)

fishdata = pd.read_csv('fish_data.csv') # CSV 파일 내용을 읽어서 Dataframe 객체로 생성
print(fishdata)
fishdata.info() # Dataframe 정보 확인
print( fishdata['Species'].value_counts() )
print( fishdata['Species'].unique())
# 7개의 물고기 종류를 분류 ===> 다중분류

# Species ==> 타깃 데이터로 준비
# Weight  Length  Diagonal   Height   Width ==> 입력 데이터로 준비
print(fishdata.columns)
fish_input = fishdata[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input)

fish_target = fishdata['Species'].to_numpy()
print(fish_target)

# train / test 데이터셋으로 분리 해서 사용
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = \
    train_test_split(fish_input, fish_target, random_state=42)

# 데이터셋 스케일 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_input)
test_scaled = scaler.transform(test_input)

print(train_scaled[:5])

# 다중분류 모델 준비
from sklearn.linear_model import LogisticRegression
multi_lrmodel = LogisticRegression(multi_class='multinomial', C=20,
                                   max_iter=1000)

# 학습
multi_lrmodel.fit(train_scaled, train_target)

# 모델 성능 평가
print( "성능 acc : " , multi_lrmodel.score(test_scaled, test_target) )

# 모델 예측
print( multi_lrmodel.predict(test_scaled[:1])) # test의 첫번째 데이터 예측
print( multi_lrmodel.predict_proba(test_scaled[:1]))
print( multi_lrmodel.classes_ )