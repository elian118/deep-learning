# 1. 데이터셋 준비하고 전처리 ( numpy, pandas 패키지를 활용 )
import pandas as pd
import numpy as np
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)
# 머신러닝 모델에서 많이 활용한 데이터셋을 이미 포함하고 있음
from sklearn.datasets import load_iris  # 붓꽃 데이터 셋

iris = load_iris()  # iris 데이터셋을 dict 형태로 반환
#print(iris)
print(iris['target_names'])
print(iris['feature_names'])

# 사전의 데이터를 추출해서 pandas의 데이터프레임 객체로 구성
iris_df = pd.DataFrame( np.column_stack( [ iris['data'], iris['target'] ]) ,
                        columns=['sepal_len','sepal_wd','petal_len','petal_wd','target'])
print(iris_df.head()) # 머리부터 5개만
print("="*50)
# print(iris_df.tail()) # 끝에서부터 5개만
# print("="*50)
# print(iris_df.sample(5)) # 랜덤 추출

# 4개 특성중 petal_len, petal_wd 만 학습 데이터의 특성으로 추출해서 준비
x_traindata = iris_df[ ['petal_len','petal_wd'] ]  # 특정 컬럼을 선택 추출
print(x_traindata)
y_traintarget = iris_df[['target']]
print(y_traintarget)

# 전체 150개 데이터를 섞으면서 train / test 로 데이터셋을 분리
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target = \
    train_test_split(x_traindata, y_traintarget, test_size=0.3, random_state=0)

print(len(train_input))
print(len(test_input))
print(train_input[:5])
print(train_target[:5])

# 2. 모델 준비
from sklearn import svm
c = 1
g = 0.5
svm_model = svm.SVC(C=c, kernel='rbf', gamma=g)  # SVM 모델 생성

#print(train_target.values.ravel()) # ravel() ==> 2차원을 1차원으로 펼쳐라!!
# 3. 모델 학습
svm_model.fit(train_input, train_target.values.ravel())
# 4. 모델 평가 및 모델 저장
from sklearn.metrics import accuracy_score
test_pred = svm_model.predict(test_input) # 테스트 데이터 대한 예측값
print("acc : ", accuracy_score(test_target.values.ravel(),test_pred)) # 97% 정확도
# 5. 모델 로딩 후 예측 (추론)

pred = svm_model.predict( test_input[:10] )
print(pred) # 2
print(test_target.values[:10])

