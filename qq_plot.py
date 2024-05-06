import scipy.stats as stats
import matplotlib.pyplot as plt

import sample_data as samp_data


data = samp_data.create_sample_normal_data(seed=0, loc=0, scale=1.0, size=1000)
stats.probplot(data, dist="norm", plot=plt)
plt.title("Normal QQ Plot")
plt.xlabel("Theoretical Quantiles")  # 理論分位数
plt.ylabel("Ordered Values")
plt.show()

exponential_data = samp_data.create_sample_exponential(seed=0, scale=1, size=1000)
stats.probplot(exponential_data, dist="norm", plot=plt)
plt.title("Normal QQ Plot to Exponential data")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Values")

plt.show()
