import itertools
import math
import numpy as np

def Valuation_Space_Discretization(m: int, epsilon:float, bias: int) -> list:
    """
    对应论文中的 Algorithm 1，用来构建离散化的估值空间。

    参数：
    - epsilon: 近似精度参数
    """

    assert epsilon > 0, "Epsilon must be positive."

    W = set()
    Z = []

    # 构建 Z
    i_max = math.ceil(math.log(1 / epsilon) / math.log(1 + epsilon))
    for i in range(0, i_max + 1):
        Z.append(epsilon * (1 + epsilon) ** i)
    
    # 构建 W
    k_max = math.ceil((2+epsilon)*m)
    for i in range(bias, i_max):
        for k in range(1, k_max + 1):
            x = Z[i] + Z[i] * epsilon * k / m
            if x <= 1.0:
                W.add(x)

    if 0 not in W:
        W.add(0)
    
    # 排序并转换为列表
    W = sorted(list(W))
    
    return W

def Amount_interval_Discretization(N: int, m: int, J: float, epsilon: float) -> list:
    """
    对应论文中 Algorithm 2 的离散化数据点集合 N_D。

    参数：
    - N: 数据点总数
    - m: 买家类型总数
    - J: 收益递减常数
    - epsilon: 近似精度参数
    """
    Q = set()
    Y = []
    N_D = set()

    # 构建 Y
    i_max = math.ceil(math.log(N * epsilon**2/ (2 * J * m)) / math.log(1 + epsilon**2))
    for i in range(0, i_max + 1):
        Y.append(math.floor((2 * J * m) / (epsilon**2) * (1 + epsilon**2) ** i))
    
    # 构建 Q    
    k_max = math.floor(2 * J * m)
    for i in range(1, len(Y)):
        for k in range(k_max + 1):
            x = math.floor(Y[i] + Y[i] * epsilon**2 * k / (2 * J * m))
            if x <= N:
                Q.add(x)
    
    # 构建 N_D
    N_D = Q
    for i in range (1, math.floor(2 * J * m / epsilon**2) + 1):
        N_D.add(i) if i <= N else None
    
    # 排序并转换为列表
    N_D = sorted(list(N_D))

    return N_D

def Price_Curves_Generation(N: int, m: int, J: float, epsilon: float, sample_nums: int) -> list:
    """
    对应论文Algorithm 2 中的价格曲线生成。

    参数：
    - N: 数据点总数
    - m: 买家类型总数
    - J: 收益递减常数
    - epsilon: 近似精度参数
    """
    # 获取离散化数据点集 (已排序)
    N_D = Amount_interval_Discretization(N, m, J, epsilon)
    last_point = N_D[-1]
    
    # 获取离散化估值空间 (已排序)
    W = Valuation_Space_Discretization(m, epsilon, bias=1)

    price_curves = []

    if sample_nums == 0:
        # 候选断点 (排除 0 和最后一个点)
        candidates = [x for x in N_D if 0 < x < last_point]
        N_array = np.array(range(N + 1))
        
        # t = 1: 常数价格曲线
        for val in W:
            price_curves.append([val] * (N + 1))

        # t >= 2: 分段价格曲线
        for t in range(2, m + 1):
            # 生成断点组合 (t - 1个实际断点)
            for break_points in itertools.combinations(candidates, t - 1):
                breaks = list(break_points)
                breaks.append(N + 1)  # 添加虚拟结束断点
                breaks_arr = np.array(breaks)
                
                # 生成非递减估值序列
                for val_seq in itertools.combinations(W, t):
                    indices = np.searchsorted(breaks_arr, N_array, side='right')
                    curve = np.take(val_seq, indices).tolist()
                    price_curves.append(curve)
    else:
        # 使用采样方法生成价格曲线
        candidates = [x for x in N_D if 0 < x < last_point]
        N_array = np.array(range(N + 1))
        for _ in range(sample_nums):
            # 随机选择断点
            num_breaks = np.random.randint(1, m + 1)
            if num_breaks == 1:
                breaks = np.array([N + 1])
            else:
                breaks = np.sort(np.random.choice(candidates, num_breaks - 1, replace=False))
                breaks = np.append(breaks, N + 1)

            # 随机生成一个非递减估值序列
            val_seq = sorted(np.random.choice(W, num_breaks, replace=True))
            indices = np.searchsorted(breaks, N_array, side='right')
            curve = np.take(val_seq, indices).tolist()
            price_curves.append(curve)

    return price_curves