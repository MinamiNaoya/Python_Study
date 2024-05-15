# カプランマイヤー推定量
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_dd

data = load_dd()
# print(data.head())
kmf = KaplanMeierFitter()

T = data["duration"]
E = data["observed"]

kmf.fit(T, event_observed=E)
kmf.survival_function_.plot()
plt.title("Survival function of political regimes")


# 信頼区間とKM推定量のプロット
kmf.plot()
plt.show()