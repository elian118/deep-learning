import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

from consts import classes

(train_input, train_target), (test_input, test_target) = fashion_mnist.load_data() # 28 * 28 이미지 샘플

print(len(train_input), len(test_input)) # 60000 10000 # 총 70000개 이미지 제공
# print(train_input)
# 총 10개 클래스 분류 데이터
print(np.unique(train_target, return_counts=True)) # 카테고리별로 각각 6000개의 이미지가 0~1로 지정돼 있음
print(train_target[0])
print(classes[train_target[0]])

print(train_input[0].shape)
plt.imshow(train_input[0], cmap='gray')
plt.show()

# import matplotlib.pyplot as plt
# img = plt.imread('./test_images/airplane.jpg')
# print(img)
# print(img.shape)
# plt.imshow(img)
# plt.show()