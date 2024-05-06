import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

rng = np.random.RandomState(1)


def generate_data(n_samples=30):
    x_min, x_max = -3, 3
    x = rng.uniform(x_min, x_max, size=n_samples)  # 一様分布を生成する。
    noise = 4.0 * rng.randn(n_samples)  # 標準正規分布に従う乱数を生成、標準偏差4, 平均0の正規分布にしている。
    y = x**3 - 0.5 * (x + 1) ** 2 + noise
    y /= y.std()
    data_train = pd.DataFrame(x, columns=["Feature"])
    data_test = pd.DataFrame(
        np.linspace(x_max, x_min, num=300), columns=["Feature"]
    )
    target_train = pd.Series(y, name="Target")
    
    return data_train, data_test, target_train

    
data_train, data_test, target_train = generate_data(n_samples=30)
sns.scatterplot(
    x=data_train["Feature"], y=target_train, color="black", alpha=0.5
)
_ = plt.title("Synthetic regression dataset")


tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(data_train, target_train)
y_pred = tree.predict(data_test)

sns.scatterplot(
    x=data_train["Feature"], y=target_train, color="black", alpha=0.5
)

plt.plot(data_test["Feature"], y_pred, label="Fitted tree")
_ = plt.title("Predictions by a single decision tree")


from sklearn.ensemble import BaggingRegressor

bagged_trees = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
)
_ = bagged_trees.fit(data_train, target_train)