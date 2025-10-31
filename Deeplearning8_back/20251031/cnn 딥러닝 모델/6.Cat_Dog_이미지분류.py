import os
print(os.getcwd())

train_dir = './cnn_cats_and_dogs_dataset/train'
test_dir = './cnn_cats_and_dogs_dataset/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 증강기법 설정
train_image_generator = ImageDataGenerator( rescale= 1.0/255.,  # 스케일 조정
                                            rotation_range=90)   # 회전
 # 검증데이터는 스케일 만 조정
test_image_generator = ImageDataGenerator( rescale= 1.0/255. )

train_data_gen = train_image_generator.flow_from_directory(train_dir,
                                  batch_size=20,
                                  class_mode='binary',
                                  target_size=(150,150))  # ==> 모델 입력 shape 설정

test_data_gen = test_image_generator.flow_from_directory(test_dir,
                                                         batch_size=20,
                                                         class_mode='binary',
                                                         target_size=(150,150))



#
# # import numpy as np
# # imgs, labels = [], []
# # for i in range(2):
# #     a, b = train_data_gen.next() # 20 batch * 2 ==> 총 40개
# #     imgs.extend(a)
# #     labels.extend(b)
# # #print(np.asarray(imgs))
# # print(np.asarray(labels))
# # print(len(np.asarray(labels)))
#

# Conv 신경망 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


 #  모델저장, 조기종료 콜백 추가
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint_cb = ModelCheckpoint('best_catdog_model.h5',save_best_only=True)
early_stopping_cb = EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_data_gen,
          validation_data=test_data_gen,
          steps_per_epoch=200, epochs=50, validation_steps=10,
          verbose=1, callbacks=[checkpoint_cb, early_stopping_cb])