import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정 -> GPU 없을때 뜨는 자잘한 경고 무시

import cv2
# img 내용 전체를 생략 없이 보려면 코드 활성
import numpy as np
# np.set_printoptions(threshold=np.inf)
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from consts import classes

# 파일 변경해가며 테스트해보기
img = cv2.imread('./test_images/sweater_3.jpg', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
print(img_resize.shape)

# 색상 반전
img_reverted = cv2.bitwise_not(img_resize)

cv2.imshow('sandal', img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
print(img_reverted.shape) # (28, 28) => (28, 28, 1) 변환 필요
new_img = img_reverted.reshape(1, 28, 28, 1) / 255.0

newmodel = load_model('mnistbestmodel.h5')

pred = newmodel.predict(new_img[:1])
final_pred = np.argmax(pred[0])
print('# 예측값')
print(classes[final_pred])