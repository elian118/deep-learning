# 1. 데이터셋 준비하고 전처리 (numpy, pandas 패키지 활용)
import pandas as pd
import numpy as np
# 판다스 출력 옵션 조정
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 1000)
# 머신러닝 모델에서 많이 활용한 데이터셋을 이미 포함하고 있음
from sklearn.datasets import load_iris # 붓꽃 데이터셋 가져오기

iris = load_iris() # iris 데이터셋을 dict 형태로 반환
# print(iris)
# print(iris['target_names'])
# print(iris['feature_names'])
# print(iris['target'])

# 사전의 데이터를 추출해서 pandas의 데이터 프레임 객체로 재구성
iris_df = pd.DataFrame(
    np.column_stack([iris['data'], iris['target']]),
    columns=['sepal_len', 'sepal_wd', 'petal_len', 'petal_wd', 'target'])
# print("=" * 53)
# print(iris_df.head()) # 머리부터 5개만
# print("=" * 53)
# print(iris_df.tail()) # 끝에서부터 5개만
# print("=" * 53)
# print(iris_df.sample(5)) # 무작위로 샘플 5개만 추출

# 4개 특성 중 petal_len, petal_wd만 학습 데이터 특성으로 추출해 준비
x_traindata = iris_df[['petal_len', 'petal_wd']] # 특정 컬럼만 선택 추출
# print("=" * 53)
# print(x_traindata.head())
y_traintarget = iris_df[['target']]
# print("=" * 53)
# print(y_traintarget.head())

# 전체 150개 데이터를 섞으면서 train / test 로 데이터셋 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = \
    train_test_split(x_traindata, y_traintarget, test_size=0.3, random_state=0)

# 데이터셋 분리 결과 확인
# print("=" * 53)
# print(len(train_input))
# print(len(test_input))
# 인덱스 순서 뒤섞였으면 random_state가 잘 적용된 것임
# print(train_input[:5])
# print(train_target[:5])

# 2. 모델 준비 -> Support Vector Machine
from sklearn import svm
c = 1
g = 0.5
svm_model = svm.SVC(C=c, kernel='rbf', gamma=g) # SVM 모델 생성

# 3. 모델 학습
# print(train_target.values.ravel()) # ravel() ==> 2차원을 1차원(기본설정)으로 펼쳐라!!
svm_model.fit(train_input, train_target.values.ravel())

# 4. 모델 평가 및 모델 저장
from sklearn.metrics import accuracy_score
test_pred = svm_model.predict(test_input) # 테스트 데이터에 대한 예측값
print("acc: ", accuracy_score(test_target.values.ravel(), test_pred))

# 5. 모델 로딩 후 예측 (추론) -> 실무에선 추론 코드를 별도 파일로 분리

pred = svm_model.predict(test_input[:1])
print(pred) # 2
print(test_target.values[0]) # 2