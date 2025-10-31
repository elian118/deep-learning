import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.datasets import load_iris # 붓꽃 데이터셋 가져오기
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

model = load_model('irisbestmodel.h5')
model.summary()

# 새 데이터 생성
new_test_input = pd.DataFrame(
    {'sepal_len': [3.5, 4.6, 3.8, 3.3, 5.5],
     'sepal_wd': [7.8, 1.7, 6.6, 1.4, 4.2],
     'petal_len': [1.5, 5.1, 1.2, 3.7, 1.6],
     'petal_wd': [4.2, 4.1, 3.7, 2.4, 5.2]})
print(new_test_input)

iris = load_iris()
# print(iris['data'])
print('# 품종')
target_names = iris['target_names']
print(target_names)
# 예측 -> ['setosa' 'versicolor' 'virginica']
pred = model.predict(new_test_input)
print('# 예측값 출력')
print(pred)
print('# 품종 예측')
for i in range(len(pred)):
    print(f"예측: {target_names[np.argmax(pred[i])]}")