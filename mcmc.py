import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform
from scipy import optimize


# 参考：https://qiita.com/YusukeOhnishi/items/d49ecb76a0d9214ca2cc
def monte():
    # 試行回数
    N_monte_list = np.arange(1000, 500000 + 1, 1000)

    sampling_rate_list = []
    for N_monte in N_monte_list:
        # 乱数を生成
        x_data = np.random.rand(N_monte)
        y_data = np.random.rand(N_monte)
        # サンプリングされた点が単位円の内部に落ちる確率
        sampling_rate = np.sum((x_data**2 + y_data**2) < 1) / N_monte
        sampling_rate_list.append(sampling_rate)
        # 円の面積 = サンプリングレート × πで求めることが可能。
        # そのため、円の半径が1のときはサンプリングレート＝面積となる。（πの推定値として使える。）
    plt.figure(figsize=(10, 6))
    plt.plot(N_monte_list, sampling_rate_list, label='Sampling Rate')
    plt.xlabel("Number of Samples")
    plt.ylabel("Sampling Rate")
    plt.title("Sampling Rate vs Number of Samples")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
def rejection_sampling(seed):
    plt.style.use("ggplot")
    np.random.seed(seed)
    
    a,b = 1.5, 2
    x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 1000)
    
    p = beta(a, b).ppf
    
    res = optimize.fmin(lambda x: -p(x), 0.3)
    y_max = p(res)
    