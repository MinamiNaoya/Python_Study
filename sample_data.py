import numpy as np


# 正規分布に従うサンプルデータを作成する
def create_sample_normal_data(seed, loc, scale, size):
    """_summary_

    Args:
        seed : 乱数シード
        loc : 正規分布の中心(平均)
        scale : 正規分布の標準偏差
        size : 乱数のサンプル数

    Returns:
        data : 生成されたデータ
    """
    np.random.seed(seed)
    data = np.random.normal(loc, scale, size)
    return data


# 指数分布に従うサンプルデータを作成する。
def create_sample_exponential(seed, scale, size):
    np.random.seed(seed)
    data = np.random.exponential(scale, size)
    return data


# 一様分布に従う乱数生成
def create_sample_random_data(seed, rows, columns):
    np.random.seed(seed)
    data = np.random.rand(rows, columns)
    return data
    