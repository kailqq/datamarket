import numpy as np

def valuation_function(buyer_type: int, n: int, N: int) -> float:
    """
    估值函数接口 v_i(n)。
    返回买家类型 buyer_type 对 n 个数据点的估值。
    
    参数:
    - buyer_type: 买家类型索引 (0, 1, ..., m-1)
    - n: 数据点数量
    - N: 数据点总量
    """
    # 动态生成买家类型参数，而不是硬编码
    # 基础速率：生成不同的收益递减速率
    # 使用buyer_type来生成差异化的参数
    np.random.seed(42 + buyer_type)  # 确保每个类型的参数固定但不同

    base_rate = 0.03 + (buyer_type + 1) * 0.02 + np.random.uniform(-0.01, 0.01)
    
    max_value = 0.6 + (buyer_type % 3) * 0.2 + np.random.uniform(-0.1, 0.1)
    
    base_rate = max(0.01, min(0.25, base_rate))
    max_value = max(0.5, min(1.5, max_value))
    
    return max_value * (1 - np.exp(-base_rate * n))


def generate_curves_algorithm2(N: int, m: int, J: float, epsilon: float, num_samples: int) -> list:
    """
    根据论文中的算法2，生成一个离散化的价格曲线集合。
    本函数通过随机采样生成一个子集。

    参数:
    - N: 数据点总量
    - m: 买家类型总数
    - J: 收益递减常数
    - epsilon: 近似精度参数
    - num_samples: 要生成的价格曲线样本数量
    """
    
    # 1. 构建数据空间网格 N_D (来自算法2)
    N_D = set()
    # 阈值以下的n不离散化
    threshold_n = int((2 * J * m) / (epsilon**2))
    for n in range(1, min(N + 1, threshold_n + 1)):
        N_D.add(n)

    # 阈值以上的n进行对数离散化
    if threshold_n < N:
        # 计算Y_i点
        y_points = []
        curr_y = threshold_n
        while curr_y < N:
            y_points.append(int(curr_y))
            curr_y *= (1 + epsilon**2)
        y_points.append(N)
        
        # 在Y_i之间插值
        for i in range(len(y_points) - 1):
            y_start, y_end = y_points[i], y_points[i+1]
            N_D.add(y_start)
            # 根据论文公式，在区间内插入约2Jm个点
            num_inner_points = int(2 * J * m)
            if num_inner_points > 1:
                inner_points = np.linspace(y_start, y_end, num_inner_points, dtype=int)
                for p in inner_points:
                    N_D.add(p)
            N_D.add(y_end)
            
    # 排序并转换为列表
    N_D = sorted(list(N_D))
    if 0 not in N_D:
        N_D.insert(0,0)


    # 2. 构建估值空间网格 W (来自算法1)
    W = set()
    # 计算Z_i点
    z_points = []
    curr_z = epsilon
    while curr_z < 1.0:
        z_points.append(curr_z)
        curr_z *= (1 + epsilon)
    z_points.append(1.0)
    
    # 在Z_i之间插值
    for i in range(len(z_points) - 1):
        z_start, z_end = z_points[i], z_points[i+1]
        gap = z_start * epsilon / m
        if gap > 0:
            num_steps = int((z_end - z_start) / gap)
            inner_points = np.linspace(z_start, z_end, num_steps)
            for p in inner_points:
                W.add(p)
    W = sorted(list(W))
    if 0 not in W:
        W.insert(0, 0)
    
    # 3.采样生成 m-阶梯 价格曲线
    curves = []
    # 首先添加算法3第一轮必须的"免费"价格曲线
    curves.append(np.zeros(N + 1))
    
    for _ in range(num_samples - 1): # -1因为已经有一条免费曲线
        # 随机决定阶梯数 k (1 到 m)
        k = np.random.randint(1, m + 1)
        
        # 从N_D中随机选择k个跳变点（确保不超过可用数量）
        available_points = [n for n in N_D if n > 0 and n <= N]
        if len(available_points) < k:
            k = len(available_points)
        
        if k > 0:
            jump_points_n = sorted(np.random.choice(available_points, k, replace=False))
            
            available_prices = [w for w in W if w >= 0]
            if len(available_prices) < k:
                # 如果价格水平不足，填充一些随机值
                additional_prices = np.random.uniform(0, 1, k - len(available_prices))
                available_prices.extend(additional_prices)
            
            price_levels = sorted(np.random.choice(available_prices, k, replace=False))
            p_curve = np.zeros(N + 1)
            for i in range(k):
                if i == 0:
                    # 第一段：从0到第一个跳变点
                    p_curve[0:jump_points_n[i] + 1] = price_levels[i]
                else:
                    # 后续段：从上一个跳变点到当前跳变点
                    p_curve[jump_points_n[i-1] + 1:jump_points_n[i] + 1] = price_levels[i]
            if jump_points_n[-1] < N:
                p_curve[jump_points_n[-1] + 1:] = price_levels[-1]
        else:
            # 如果k=0，使用一个常数价格曲线
            constant_price = np.random.choice(W)
            p_curve = np.full(N + 1, constant_price)
            
        curves.append(p_curve)
        
    return curves


def precompute_outcomes(price_curves: list, m_types: int, N_items: int):
    """
    预先计算每个(价格曲线, 买家类型)组合的购买决策和收入。
    这模拟了Agent拥有的先验知识（即知道v_i函数）。
    """
    num_curves = len(price_curves)
    # 记录每种组合产生的收入
    revenues = np.zeros((num_curves, m_types))
    # 记录每种组合的购买量，用于确定S_t
    purchases = np.zeros((num_curves, m_types), dtype=int)

    for i, p_curve in enumerate(price_curves):
        for j in range(m_types):
            # 计算所有n的效用
            utils = [valuation_function(j, n, N_items) - p_curve[n] for n in range(N_items + 1)]
            max_util = np.max(utils)
            
            if max_util < 0:
                n_purchase = 0
            else:
                # 找到所有效用最大的n，并根据论文规则选择最大的那个
                optimal_n_indices = np.where(np.isclose(utils, max_util))[0]
                n_purchase = optimal_n_indices[-1]
            
            purchases[i, j] = n_purchase
            revenues[i, j] = p_curve[n_purchase]
            
    return revenues, purchases
