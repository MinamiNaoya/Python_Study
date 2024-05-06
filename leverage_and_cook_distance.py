import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import sample_data as samp_data
# y = 2X(Xの100×2行列の1列目) + 3X(Xの100行2列の2列目) + ε 
X = samp_data.create_sample_random_data(seed=0, rows=100, columns=2)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(scale=0.1, size=100)  # ターゲット

# 線形回帰モデルを構築
X_with_intercept = sm.add_constant(X)  # 定数項を追加
model = sm.OLS(y, X_with_intercept)
results = model.fit()

# レバレッジを計算
leverage = results.get_influence().hat_matrix_diag

# Cook's距離を計算
cooks_distance = results.get_influence().cooks_distance[0]

# レバレッジとCook's距離のプロット
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(range(len(leverage)), leverage, marker='o', color='blue')
plt.title('Leverage')
plt.xlabel('Data Index')
plt.ylabel('Leverage')

plt.subplot(1, 2, 2)
plt.scatter(range(len(cooks_distance)), cooks_distance, marker='o', color='green')
plt.title("Cook's Distance")
plt.xlabel('Data Index')
plt.ylabel("Cook's Distance")

plt.tight_layout()
plt.show()

