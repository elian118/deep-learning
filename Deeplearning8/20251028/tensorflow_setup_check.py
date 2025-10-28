import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 최소 경고 수준으로 설정
import numpy as np
import pandas as pd
import tensorflow as tf

a = tf.constant([1,2,3,4])
print(a)  # 결과 출력되면 성공