from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 정답
y_true = [0,1,1,0,1,1,1,1,0,0,2,2,3,2,3]
y_pred = [0,1,0,0,1,1,1,1,0,0,2,2,3,2,2]

print( confusion_matrix(y_true, y_pred) )
print( classification_report(y_true, y_pred) )
# shift + F10 : 실행명령
import os

print( os.getcwd() )
