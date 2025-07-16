import numpy as np
import visualization
from utils import valuation_function, generate_curves_algorithm2, precompute_outcomes


class Config:
    def __init__(self):
        self.M_TYPES = 10           # 买家类型数量
        self.N_ITEMS = 100          # 数据点总数
        self.T_HORIZON = self.M_TYPES * 10000      # 总回合数
        
        # 自动生成真实的买家类型分布
        self.Q_TRUE = self._generate_true_distribution(self.M_TYPES)
        
        self.J = 2.0                   # 收益递减常数
        self.epsilon = 0.1             # 近似精度参数
        self.num_samples = max(7, self.M_TYPES + 5)  # 生成的价格曲线数量
    
    def _generate_true_distribution(self, m_types):
        """
        自动生成真实的买家类型分布
        """
        if m_types == 1:
            return np.array([1.0])
        elif m_types == 2:
            return np.array([0.6, 0.4])
        elif m_types == 3:
            return np.array([0.6, 0.3, 0.1])
        elif m_types == 4:
            return np.array([0.5, 0.3, 0.15, 0.05])
        else:
            # 对于更多类型，使用高斯分布
            distribution = np.random.normal(loc=0.5, scale=0.1, size=m_types)
            distribution = np.abs(distribution)  # 确保所有值为正
            distribution /= np.sum(distribution)  # 归一化为概率分布
            
            return distribution
    
    def print_config(self):
        """
        打印配置信息
        """
        print(f"   参数设置:")
        print(f"   买家类型数量 (M_TYPES) = {self.M_TYPES}")
        print(f"   数据点总数 (N_ITEMS) = {self.N_ITEMS}")
        print(f"   总回合数 (T_HORIZON) = {self.T_HORIZON}")
        print(f"   真实买家分布 (Q_TRUE) = {self.Q_TRUE}")
        print(f"   收益递减常数 (J) = {self.J}")
        print(f"   近似精度参数 (epsilon) = {self.epsilon}")
        print(f"   生成曲线数量 (num_samples) = {self.num_samples}")
        print()
        
        # 验证分布是否有效
        if abs(np.sum(self.Q_TRUE) - 1.0) > 1e-10:
            print("真实分布概率和不等于1")
        else:
            print("真实分布有效")

class UCBAgent:
    """
    实现了论文中算法3的UCB。
    """
    def __init__(self, m_types: int, T_horizon: int, precomputed_revenues, precomputed_purchases):
        """
        初始化Agent的状态。
        
        参数:
        - m_types: 买家类型总数 (m)
        - T_horizon: 总轮数 (T)
        - precomputed_revenues: 预计算的收入矩阵
        - precomputed_purchases: 预计算的购买量矩阵
        """
        self.m = m_types
        self.T = T_horizon
        self.revenues = precomputed_revenues
        self.purchases = precomputed_purchases
        self.num_curves = precomputed_revenues.shape[0]

        # Agent内部状态
        self.t = 1  # 当前回合数，从1开始
        # T_{i,t}: 类型i被"探索"的次数
        self.T_i = np.zeros(m_types)
        # N_i: 类型i被实际观察到的次数
        self.N_i = np.zeros(m_types)
        
        # 用于可视化的历史记录
        self.history = {
            'round': [],
            'action': [],
            'revenue': [],
            'cumulative_revenue': [],
            'buyer_type': [],
            'q_estimates': [],
            'confidence_bounds': [],
            'ucb_values': [],
            'T_i_history': [],
            'N_i_history': []
        }

    def choose_action(self) -> int:
        """
        根据UCB规则选择一个价格曲线的索引。
        """
        # 算法规定，第一轮免费，以确保获得反馈
        if self.t == 1:
            # 第一轮也需要记录值，即使是初始值
            q_bar = np.zeros(self.m)
            confidence_bonus = np.zeros(self.m)
            rev_hat = np.zeros(self.num_curves)
            
            self.history['q_estimates'].append(q_bar.copy())
            self.history['confidence_bounds'].append(confidence_bonus.copy())
            self.history['ucb_values'].append(rev_hat.copy())
            
            return 0  # 假设索引0是全零价格曲线

        # 1. 计算对q的经验估计 q_bar
        #    为避免除以0，当T_i为0时，q_bar也为0
        T_i_safe = np.maximum(self.T_i, 1) # 防止除零
        q_bar = self.N_i / T_i_safe

        # 2. 计算置信度奖励项
        confidence_bonus = np.sqrt(np.log(self.T) / T_i_safe)

        # 3. 构造乐观的q估计 q_hat
        q_hat = q_bar + confidence_bonus

        # 4. 计算每个价格曲线的期望收入的UCB (rev_hat)
        #    这是论文公式(7)的实现
        #    rev_hat(p) = sum(q_hat_i * revenue(p, i)) for i in 1..m
        rev_hat = self.revenues @ q_hat  
        
        # 记录UCB值用于可视化
        self.history['q_estimates'].append(q_bar.copy())
        self.history['confidence_bounds'].append(confidence_bonus.copy())
        self.history['ucb_values'].append(rev_hat.copy())
        
        # 5. 选择具有最高UCB收入的曲线
        chosen_curve_idx = np.argmax(rev_hat)
        return chosen_curve_idx

    def update(self, chosen_curve_idx: int, feedback_type: int or None):
        """
        根据环境反馈更新内部状态。
        
        参数:
        - chosen_curve_idx: 本轮选择的价格曲线的索引
        - feedback_type: 观察到的买家类型索引，如果未购买则为None
        """
        # 1. 更新探索次数 T_i
        #    S_t 是本轮价格下会购买的买家类型集合
        #    即能够吸引哪些买家
        S_t = np.where(self.purchases[chosen_curve_idx, :] > 0)[0]
        self.T_i[S_t] += 1

        
        # 2. 如果有购买，更新观察次数 N_i
        #   即此轮出现的买家刚好在S_t中，即买了，那么给对应类型的N_i加1
        revenue_this_round = 0
        if feedback_type is not None:
            self.N_i[feedback_type] += 1
            revenue_this_round = self.revenues[chosen_curve_idx, feedback_type]
        
        # 记录历史数据
        self.history['round'].append(self.t)
        self.history['action'].append(chosen_curve_idx)
        self.history['revenue'].append(revenue_this_round)
        self.history['cumulative_revenue'].append(
            self.history['cumulative_revenue'][-1] + revenue_this_round if self.history['cumulative_revenue'] else revenue_this_round
        )
        self.history['buyer_type'].append(feedback_type)
        self.history['T_i_history'].append(self.T_i.copy())
        self.history['N_i_history'].append(self.N_i.copy())
            
        # 进入下一轮
        self.t += 1

# ==============================================================================
# 3. 综合运行函数
# ==============================================================================

def run_with_visualization(M_TYPES=3, N_ITEMS=100, T_HORIZON=5000, Q_TRUE=None,J=2.0,epsilon=0.1,num_samples=7):
    """
    运行UCB算法并生成所有可视化
    """
    
    print("🚀 开始UCB算法可视化演示")
    print("=" * 50)
    
    print("\n📊 1. 准备数据...")
    price_curves = generate_curves_algorithm2(N_ITEMS, M_TYPES, J, epsilon, num_samples)
    revenues, purchases = precompute_outcomes(price_curves, M_TYPES, N_ITEMS)
    
    # 2. 初始化并运行UCB Agent
    print("\n🤖 2. 初始化UCB Agent...")
    agent = UCBAgent(M_TYPES, T_HORIZON, revenues, purchases)
    
    print(f"\n🎯 3. 开始模拟运行 (T={T_HORIZON})...")
    total_revenue = 0
    
    # 用于进度显示
    progress_points = [int(T_HORIZON * p) for p in [0.1, 0.25, 0.5, 0.75, 1.0]]
    progress_idx = 0
    
    for t in range(T_HORIZON):
        # 显示进度
        if progress_idx < len(progress_points) and t >= progress_points[progress_idx]:
            print(f"进度: {int(100 * (progress_idx + 1) / len(progress_points))}% 完成")
            progress_idx += 1
        
        # 1. Agent选择动作
        action_idx = agent.choose_action()
        
        # 2. 环境生成买家
        true_type = np.random.choice(M_TYPES, p=Q_TRUE)
        
        # 3. 确定结果
        n_bought = purchases[action_idx, true_type]
        feedback = None
        
        if n_bought > 0:
            revenue_this_round = revenues[action_idx, true_type]
            total_revenue += revenue_this_round
            feedback = true_type
        
        # 4. 更新Agent
        agent.update(action_idx, feedback)
    
    print(f"\n✅ 模拟完成!")
    print(f"总收入: {total_revenue:.2f}")
    final_q_estimate = agent.N_i / np.maximum(agent.T_i, 1)
    
    estimation_error = np.abs(config.Q_TRUE - final_q_estimate)

    print(f"学习到的估计: {np.round(final_q_estimate, 3)}")
    print(f"估计误差: {np.round(estimation_error, 3)}")
    print(f"平均绝对误差: {np.mean(estimation_error):.3f}")

    # 可视化
    output_folder = None

    print("\n 生成可视化...")
    output_folder = visualization.run_visualization_suite(
        M_TYPES, N_ITEMS, T_HORIZON, Q_TRUE, 
        price_curves, valuation_function, revenues, purchases, agent
    )

    return agent, output_folder

# ============================================================================================================================================================
if __name__ == '__main__':    
    config = Config()
    print(f"\n{'='*60}")
    print(f"测试买家类型数量: {config.M_TYPES}")
    print(f"{'='*60}")
    
    config.print_config()
    
    # 运行完整的可视化演示
    agent, output_folder = run_with_visualization(
        config.M_TYPES, config.N_ITEMS, 
        config.T_HORIZON, config.Q_TRUE, 
        config.J, config.epsilon, config.num_samples)
    
    print("-" * 30)
    
    # 计算regret
    best_curve_revenues = np.max(agent.revenues @ config.Q_TRUE)
    theoretical_best = best_curve_revenues * config.T_HORIZON
    actual_revenue = agent.history['cumulative_revenue'][-1]
    regret = theoretical_best - actual_revenue
    
    print(f"理论最优单轮收入: {best_curve_revenues:.3f}")
    print(f"理论最优总收入: {theoretical_best:.2f}")
    print(f"实际获得总收入: {actual_revenue:.2f}")
    print(f"总regret: {regret:.2f}")
    print(f"平均regret: {regret/config.T_HORIZON:.4f}")
    
    # 显示最终的价格曲线偏好
    action_counts = np.bincount(agent.history['action'], minlength=agent.num_curves)
    best_action = np.argmax(action_counts)
    print(f"最常选择的价格曲线: 曲线{best_action} (选择了 {action_counts[best_action]} 次)")
    
    if output_folder is not None:
        print(f"📂 请查看生成的图片文件夹: {output_folder}")
    
    print(f"\n✅ 测试完成!")
