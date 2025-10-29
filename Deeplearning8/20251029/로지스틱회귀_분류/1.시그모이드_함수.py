#  p84
import numpy as np
import matplotlib.pyplot as plt

# 소숫점의 과학적 표기법(ex. 1.00000000e+00)의 사용 억제
np.set_printoptions(suppress=True)
x = np.arange(-5, 5, 0.1)
print(x)

y = 1 / (1 + np.exp(-x)) # 시그모이드 함수 ==> 0 ~ 1 사이 값으로 반환

plt.plot(x, y)
plt.show()