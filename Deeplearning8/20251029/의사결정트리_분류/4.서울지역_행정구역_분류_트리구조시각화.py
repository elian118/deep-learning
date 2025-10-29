# p51 ~ p52
import pandas as pd
from dataset import district_dict_list, dong_dict_list

train_df = pd.DataFrame(district_dict_list)
train_df = train_df[['district','longitude','latitude','label']]
print(train_df)

test_df = pd.DataFrame(dong_dict_list)
test_df = test_df[['dong', 'longitude', 'latitude', 'label']]
print(test_df)

print(train_df['label'].value_counts())
print(test_df['label'].value_counts())

# train_df 의 longitude(경도), latitude(위도)  ==> 학습 데이터
# train_df 의 label  ==> 학습 데이터의 목표(라벨)

# test_df 의 longitude(경도), latitude(위도)  ==> 테스트 데이터
# test_df 의 label  ==> 테스트 데이터의 목표(라벨)

train_df.drop(['district'], axis=1, inplace=True)
test_df.drop(['dong'], axis=1, inplace=True)


X_train = train_df[['longitude','latitude']]
Y_train = train_df[['label']]

# 의사결정트리는 각 특징을 독립적으로 사용하기 떄문에 별다른 전처리 과정 필요 없음

X_test = test_df[['longitude','latitude']]
Y_test = test_df[['label']]

# 사이킷런 의사결정 트리 모델 학습
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
Y_encoded = le.fit_transform(Y_train.values.ravel())
print(Y_encoded)
print(Y_train.values.ravel())
print(le.classes_)

# 과대적합 회피 파라미터 설정 모델
clf = tree.DecisionTreeClassifier(
    # criterion='entropy', # 기본 'gini'
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=70).fit(X_train,Y_encoded)
#clf = tree.DecisionTreeClassifier().fit(X_train, Y_encoded) : 과대적합 모델

from sklearn.tree import plot_tree
plt.figure(figsize=(8,8))
# filled : 클래스에 맞게 노드의 색을 칠함
plot_tree(clf, filled=True, feature_names=['longitude','latitude'],
          class_names=['Gangbuk', 'Gangdong', 'Gangnam', 'Gangseo'])
plt.show()
# 파일로 저장
# plt.savefig("gini.png")