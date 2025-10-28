from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 정답
y_ture = [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 2, 2, 3, 2, 3]
# 예측
y_pred = [0, 1, 0 ,0, 1, 1, 1, 1, 0, 0, 2, 2, 3, 2, 2]

print(confusion_matrix(y_ture, y_pred))
# [[4 0 0 0]
#  [1 5 0 0]
#  [0 0 3 0]
#  [0 0 1 1]]
print(classification_report(y_ture, y_pred))
#               precision    recall  f1-score   support
#
#            0       0.80      1.00      0.89         4
#            1       1.00      0.83      0.91         6
#            2       0.75      1.00      0.86         3
#            3       1.00      0.50      0.67         2
#
#     accuracy                           0.87        15
#    macro avg       0.89      0.83      0.83        15
# weighted avg       0.90      0.87      0.86        15

import os

print(os.getcwd()) # C:\Users\USER\Desktop\hancom\deep-learning\Deeplearning8\20251028