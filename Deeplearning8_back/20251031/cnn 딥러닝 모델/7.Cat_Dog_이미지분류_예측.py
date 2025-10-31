from tensorflow.keras.models import load_model

bestcnn_model = load_model('best_catdog_model.h5')  # 앞의 저장 모델 로딩

from tensorflow.keras.preprocessing import image  # 이미지 전처리 모듈

# 임의의 예측 데이터 로딩
# test_dog.jpg 인터넷에서 강아지 이미지 다운받아서 해당 위치에 저장
img = image.load_img('./test_dog.jpg', target_size=(150,150))

img_arr = image.img_to_array(img) / 255.0  # 데이터 정규화
print(img_arr)
print(img_arr.shape)


# 새로운 이미지 예측
pred = bestcnn_model.predict(img_arr.reshape(1,150,150,3),batch_size=1)
print(pred)

import numpy as np
pre_result = np.where(pred[0] > 0.5, 1, 0)   # cats label : 0, dogs label : 1
print(pre_result)        #   [1]

cat_dog_classnames = np.array(['cat','dog'])

print(cat_dog_classnames[pre_result])