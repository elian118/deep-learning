from visualize import display_decison_surface
import pandas as pd
from dataset import district_dict_list, dong_dict_list

train_df = pd.DataFrame(district_dict_list)
train_df = train_df[['district','longitude','latitude','label']] # 컬럼 데이터 순서 변경
print(train_df)

test_df = pd.DataFrame(dong_dict_list)
test_df = test_df[['dong','longitude','latitude','label']] # 컬럼 데이터 순서 변경
print(test_df)

# 컬럼 데이터 종류와 개수를 반환하는 메서드
print(train_df['label'].value_counts())
print(test_df['label'].value_counts())

# inplace=True: 사본 객체 만들지 말고 원본에 직접 반영
# Dataframe 객체에 특정 컬럼(axis=1) 데이터를 삭제
train_df.drop(['district'], axis=1, inplace=True)
print(train_df)

train_input = train_df[['longitude', 'latitude']]
train_target = train_df[['label']]

test_input = test_df[['longitude', 'latitude']]
test_target = test_df[['label']]

# p49
from sklearn import tree # 의사결정트리 모델
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing # sklearn 지원하는 데이터 전처리 라이브러리(패키지)

# 데이터 가공 예시 -> 여기선 안 씀.. 이런 것도 있다는 것 정도로만 알아두기
# train_target['label'] = train_target['label'].map({'Gangseo': 1, 'Gangnam': 2})
# print(train_target)

le = preprocessing.LabelEncoder() # 학습 비교를 위해 문자열 자료를 구분할 수 있는 각 숫자로 변환하는 전처리 함수
train_target_encode = le.fit_transform(train_target.values.ravel())
print('=' * 55)
print(train_target.values.ravel())
print('=' * 55)
print(le.classes_) # 알파벳 순으로 인덱스가 정렬돼 있음
print(train_target_encode) # 따라서, train_target_encode는 le.classes_ 근거로 각 인덱스 값으로 치환된 값임

# 의사결정트리 모델 준비 -> 각 하이팟 파라미터에 대한 설명은 50p 아래 박스 참고
trmodel = tree.DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=35) # random_state 값만 입력하면 과대적합 발생

# 의사결정트리 모델 학습
clf = trmodel.fit(train_input, train_target_encode)

# 임의의 데이터 예측
pred = trmodel.predict([[125, 36], [127, 38]])
print('=' * 55)
print(pred) # [3, 0]
print(le.classes_[pred][0])
print(le.classes_[pred][1])
tempxy = [[127.03, 37.51]]

def display_decison_surface(clf, X,  tempxy):
    x_min = X['longitude'].min() - 0.01
    x_max = X['longitude'].max() + 0.01
    y_min = X['latitude'].min() - 0.01
    y_max = X['latitude'].max() + 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                         np.arange(y_min, y_max, 0.001))
    #print(xx)  # [[126.8395 126.8405 126.8415 ... 127.1585 127.1595 127.1605]
    #print(xx.shape)  #(237, 322)
    np.set_printoptions(threshold=np.inf)
    #print(xx.ravel())
    #print(yy.ravel())
    Z = clf.predict(np.column_stack([xx.ravel(), yy.ravel()]))
    print(Z.shape)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,7))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    # color 참조 : https://matplotlib.org/stable/gallery/color/named_colors.html
    plt.scatter(tempxy[0][0],tempxy[0][1], c='indigo', edgecolors='black', s=150)

    plt.title('Decision Surface of predict data', fontsize=16)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()

display_decison_surface(clf, train_input, tempxy)