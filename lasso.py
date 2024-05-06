import pandas as pd
import numpy as np


from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names).assign(MEDV=boston.target)

y = df.iloc[:, -1]
df = (df - df.mean()) / df.std()

x = df.iloc[:, :-1]
x = np.column_stack((np.ones(len(x)), x))

n = x.shape[0]
d = x.shape[1]
w = np.zeros(d)
r = 1.0

for _ in range(1000):
    for k in range(d):
        if k == 0:
            w[0] = (y - np.dot(x[:, 1:])).sum() / n
        
        else:
            # バイアス、更新対象の重み 以外の添え字
            _k = [i for i in range(d) if i not in [0, k]]
            # wk更新式の分子部分
            a = np.dot((y - np.dot(x[:, _k], w[_k]) - w[0]), x[:, k]).sum()
            # wk更新式の分母部分
            b = (x[:, k] ** 2).sum()

            if a > n * r:  # wkが正となるケース
                w[k] = (a - n * r) / b
            elif a < -r * n:  # wkが負となるケース
                w[k] = (a + n * r) / b
            else:  # それ以外のケース
                w[k] = 0
            
