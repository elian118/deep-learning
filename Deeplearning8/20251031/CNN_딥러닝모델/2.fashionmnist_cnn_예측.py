import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from consts import classes

# 소숫점의 과학적 표기법의 사용 억제
np.set_printoptions(suppress=True)

(train_input, train_target), (test_input, test_target) = fashion_mnist.load_data() # 28 * 28 이미지 샘플

from tensorflow.keras.models import load_model

newmodel = load_model('mnistbestmodel.h5')
# newmodel.summary()

print(test_input.shape)
# 예측을 위한 shape 변경
test_input_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
print(test_input_scaled.shape)

pred = newmodel.predict(test_input_scaled[:1])
print('# 예측결과 출력')
print(pred)
final_pred = np.argmax(pred[0])
print('# 예측값')
print(classes[final_pred])
print(test_target[0:1])

plt.imshow(test_input[0])
plt.show()