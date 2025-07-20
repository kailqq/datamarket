import os
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import Price_Curves_Generation
from collections import Counter

class FTPLAgent:
    def __init__(self, N, m, J, T, epsilon, theta, valuation, sample_nums=0):
        """
        初始化 FTPLAgent 实例
        :param N: 数据点总数
        :param m: 买家类型数量
        :param J: 收益递减常数
        :param T: 总回合数
        :param epsilon: 离散化精度参数
        :param theta: 价格曲线扰动参数
        :param valuation: 买家估值函数
        :param sample_nums: 生成的价格曲线数量
        """
        self.N = N
        self.m = m
        self.J = J
        self.T = T
        self.epsilon = epsilon
        self.theta = theta
        self.valuation = valuation
        self.sample_nums = sample_nums
        self.quantities = list(range(1, N+1))
        self.price_curves = Price_Curves_Generation(N, m, J, epsilon, sample_nums)
        self.perturbations = np.random.exponential(scale=1/theta, size=len(self.price_curves)) / 100
        
        os.makedirs('FTPL_visualization', exist_ok=True)

        # 初始化记录字典
        self.records = {
            'buyers': [],
            'history_rewards': np.zeros((self.T, len(self.price_curves))),
            'history_rewards_sum': np.zeros(len(self.price_curves)),
            'chosen_curve_indices': [],
            'chosen_curves': [],
            'payments': [],
            'purchase_nums': [],
            'rounds': [],
            'cumulative_revenue': [],
            'regret': []
        }

    def Online_Adversarial_Pricing(self):
        """
        执行在线对抗定价算法
        1. 卖家选择价格曲线
        2. 买家选择类型并进行购买
        3. 更新记录
        """
        cumulative = 0
        for t in range(self.T):
            self.Sellers_Action()
            self.Adversarial_Action()
            self.Update_Record()

            cumulative += self.records['payments'][-1]
            self.records['cumulative_revenue'].append(cumulative)
            self.records['rounds'].append(t+1)

    def Sellers_Action(self):
        """
        基于 Algorithm 4 的价格曲线选择策略
        """
        # 根据历史奖励和扰动之和选择价格曲线
        current_rewards = self.records['history_rewards_sum'] + self.perturbations
        chosen_index = np.argmax(current_rewards)
        
        # 更新记录
        self.records['chosen_curve_indices'].append(chosen_index)
        self.records['chosen_curves'].append(self.price_curves[chosen_index])
        return chosen_index

    def Adversarial_Action(self):
        """
        对抗设置，遍历所有买家，选择使得卖家收益最小的买家类型
        """
        chosen_index = self.records['chosen_curve_indices'][-1]
        
        worst_type = 0
        worst_result = (0, 0)
        min_revenue = float('inf')
        for buyer_type in range(self.m):
            num, payment = self.Optical_Purchase(self.price_curves[chosen_index], buyer_type)
            if payment < min_revenue:
                min_revenue = payment
                worst_type = buyer_type
                worst_result = (num, payment)

        # 更新记录
        self.records['buyers'].append(worst_type)
        self.records['purchase_nums'].append(worst_result[0])
        self.records['payments'].append(worst_result[1])


    def Update_Record(self):
        """
        根据 Algorithm 4 的规则更新记录
        """
        # 获取当前回合数和被选中的价格曲线索引
        t = len(self.records['buyers']) - 1
        chosen_index = self.records['chosen_curve_indices'][t]
        buyers_type = self.records['buyers'][t]
        purchase_num = self.records['purchase_nums'][t]

        # 初始化当前回合的奖励
        round_rewards = np.zeros(len(self.price_curves))

        if purchase_num > 0:
            # 如果有人购买：
            for p in range(len(self.price_curves)):
                # 按照当前买家类型更新每个价格曲线的奖励
                _, virtual_payment = self.Optical_Purchase(
                    self.price_curves[p], buyers_type
                )
                self.records['history_rewards_sum'][p] += virtual_payment
                round_rewards[p] = virtual_payment
        else:
            # 如果没有人购买：
            S_t = set()
            # 遍历所有买家类型，找到所有满足会购买的买家类型
            for i in range(self.m):
                for n in range(self.N + 1):
                    if self.valuation(i, n) >= self.price_curves[chosen_index][n]:
                        S_t.add(i)
                        break
            
            # S_t 的补集，即所有不会购买的买家类型
            S_t_c = set(range(self.m)) - S_t

            # 对于不会购买的买家类型，计算每个价格曲线的奖励，累加作为当前回合的奖励
            for p in range(len(self.price_curves)):
                r_t_p = 0.0
                for i in S_t_c:
                    _, opt_payment = self.Optical_Purchase(
                        self.price_curves[p], i
                    )
                    r_t_p += opt_payment
                self.records['history_rewards_sum'][p] += r_t_p
                round_rewards[p] = r_t_p

        # 更新历史奖励记录
        self.records['history_rewards'][t, :] = round_rewards

        # 计算遗憾
        buyers_seq = self.records['buyers']
        max_revenue = -float('inf')
        for curve in self.price_curves:
            revenue = 0
            for buyer_type in buyers_seq:
                _, payment = self.Optical_Purchase(curve, buyer_type)
                revenue += payment
            if revenue > max_revenue:
                max_revenue = revenue
        actual_revenue = np.sum(self.records['payments'])
        regret = max_revenue - actual_revenue
        self.records['regret'].append(regret)

    def Optical_Purchase(self, price_curve: list, buyers_type: int):
        """
        根据买家类型和价格曲线，计算最优购买数量和支付金额
        """

        max_util = 0
        num = 0
        payments = 0

        # 如果存在收益大于 0 的购买数量，则选择最大效用的购买数量
        for n in range(1, self.N + 1):
            val = self.valuation(buyers_type, n)
            price = price_curve[n]
            util = val - price

            # 用 >= 而不是 >，以确保在效用相等时选择较大的购买数量
            if util >= max_util:
                max_util = util
                num = n
                payments = price
        
        return num, payments
    
    def Visualization(self):
        """
        生成完整的可视化图表，用图片形式保存到 FTPL_visualization 目录。
        """
        # 确保输出目录存在
        os.makedirs('FTPL_visualization', exist_ok=True)
        
        # 1. 价格曲线选择历史
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        self.plot_price_curve_selection(ax1)
        fig1.savefig('FTPL_visualization/FTPL_price_curve_selection.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. 买家类型与支付
        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        self.plot_buyer_payments(ax2)
        fig2.savefig('FTPL_visualization/FTPL_buyer_payments.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. 累积收益
        fig3 = plt.figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        self.plot_cumulative_revenue(ax3)
        fig3.savefig('FTPL_visualization/FTPL_cumulative_revenue.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # 4. 价格曲线奖励分布
        fig4 = plt.figure(figsize=(10, 6))
        ax4 = fig4.add_subplot(111)
        self.plot_reward_distribution(ax4)
        fig4.savefig('FTPL_visualization/FTPL_reward_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)

        # 5. 买家估值函数
        fig5 = plt.figure(figsize=(10, 6))
        ax6 = fig5.add_subplot(111)
        self.plot_valuation_functions(ax6)
        fig5.savefig('FTPL_visualization/FTPL_valuation_functions.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)

    def plot_price_curve_selection(self, ax):
        """
        绘制价格曲线选择历史（分段函数），并标记每条曲线被选择的次数
        """
        # 获取所有被选中的曲线
        selected_indices = self.records['chosen_curve_indices']
        selected_curves = [self.price_curves[i] for i in selected_indices]
        
        # 统计每条曲线被选择的次数
        count_dict = Counter(selected_indices)
        
        # 为不同轮次定义颜色范围
        num_curves = len(selected_curves)
        cmap = plt.get_cmap('tab20')
        if num_curves <= 20:
            colors = [cmap(i) for i in range(num_curves)]
        else:
            colors = [cmap(i / num_curves) for i in range(num_curves)]
        
        # 绘制被选中的价格曲线（使用分段函数/阶梯图）
        for i, (curve_idx, curve) in enumerate(zip(selected_indices, selected_curves)):
            x_points = np.arange(0, self.N + 1)
            y_points = curve
            ax.step(
                x_points, y_points,
                where='post',
                linewidth=2,
                alpha=0.8 if i == len(selected_curves)-1 else 0.5,
                color=colors[i % len(colors)],
                label=f'Round {i+1}'
            )
            ax.scatter(
                x_points, y_points,
                s=40,
                color=colors[i % len(colors)],
                alpha=0.8 if i == len(selected_curves)-1 else 0.5
            )
            # 标记曲线被选择的次数
            if selected_indices.index(curve_idx) == i:
                count = count_dict[curve_idx]
                ax.text(
                    x_points[-1], y_points[-1],
                    f'×{count}',
                    color=colors[i % len(colors)],
                    fontsize=10,
                    va='bottom', ha='right', fontweight='bold'
                )
        
        # 设置图表属性
        ax.set_title('Price Curve Selection')
        ax.set_xlabel('Quantity (n)')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.set_xticks(np.arange(0, self.N + 1))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.text(0.5, -0.15, 'Step functions represent piecewise constant price curves',
            transform=ax.transAxes, ha='center', fontsize=10, color='gray')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    def plot_buyer_payments(self, ax):
        """
        两个图叠加显示：
        - 支付金额柱状图
        - 买家效益折线图
        """
        rounds = self.records['rounds']
        buyers = self.records['buyers']
        payments = self.records['payments']
        purchase_nums = self.records['purchase_nums']
        chosen_indices = self.records['chosen_curve_indices']

        # 支付金额柱状图
        bars = ax.bar(rounds, payments, color='skyblue', alpha=0.7, label='Payment')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # 购买数量
            ax.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                    f'{purchase_nums[i]}', ha='center', va='bottom', fontsize=9, color='black')

        # 买家效益折线
        utils = []
        for i in range(len(rounds)):
            buyer_type = buyers[i]
            n = purchase_nums[i]
            curve = self.price_curves[chosen_indices[i]]
            if n > 0:
                val = self.valuation(buyer_type, n)
                price = curve[n]
                util = val - price
            else:
                util = 0
            utils.append(util)

        ax2 = ax.twinx()
        ax2.plot(rounds, utils, marker='o', color='orange', label='Utility')
        ax2.set_ylabel('Utility')

        # 设置图表属性
        ax.set_title('Buyer Payments and Utility per Round')
        ax.set_xlabel('Round')
        ax.set_ylabel('Payment')
        ax.grid(axis='y')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 0.92), borderaxespad=0)
    
    def plot_cumulative_revenue(self, ax):
        """
        绘制累积收益和遗憾曲线（叠加显示）
        """
        rounds = self.records['rounds']
        cumulative_revenue = self.records['cumulative_revenue']
        regret = self.records['regret']

        ax.plot(rounds, cumulative_revenue, 'o-', markersize=6, linewidth=2, color='green', label='Cumulative Revenue')
        ax.set_xlabel('Round')
        ax.set_ylabel('Cumulative Revenue', color='green')
        ax.tick_params(axis='y', labelcolor='green')
        ax.grid(True, axis='y')

        # 标注最终收益
        final_rev = cumulative_revenue[-1]
        ax.text(self.T, final_rev, f'Total: {final_rev:.2f}', 
                ha='right', va='bottom', fontsize=10, color='green')

        # 叠加遗憾曲线
        ax2 = ax.twinx()
        ax2.plot(rounds, regret, 's--', markersize=5, linewidth=2, color='red', label='Regret')
        ax2.set_ylabel('Regret', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 标注最终遗憾
        final_regret = regret[-1]
        ax2.text(self.T, final_regret, f'Regret: {final_regret:.2f}',
                 ha='right', va='top', fontsize=10, color='red')

        # 图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        ax.set_title('Cumulative Revenue and Regret')
    
    def plot_reward_distribution(self, ax):
        """
        显示采样的 100 条价格曲线和被选中的价格曲线的奖励分布
        纵坐标按照轮数 10% ~ 100% 分十段堆叠显示
        """
        history_rewards = self.records['history_rewards']
        T, M = history_rewards.shape

        # 统计被选中的价格曲线及其被选次数
        chosen_indices = self.records['chosen_curve_indices']
        unique_chosen, counts_chosen = np.unique(chosen_indices, return_counts=True)
        chosen_count_dict = dict(zip(unique_chosen, counts_chosen))
        chosen_set = set(unique_chosen)

        # 均匀采样，选取 100 条
        num_display = min(100, M)
        all_indices = np.arange(M)
        not_chosen_indices = np.array([idx for idx in all_indices if idx not in chosen_set])
        rng = np.random.default_rng(42)
        if len(not_chosen_indices) > num_display:
            sampled = rng.choice(not_chosen_indices, size=len(not_chosen_indices), replace=False)
            step = len(sampled) / num_display
            display_indices = [sampled[int(i * step)] for i in range(num_display)]
        else:
            display_indices = not_chosen_indices.tolist()

        # 加上所有被选中的曲线
        display_indices = list(display_indices) + list(unique_chosen)

        # 从大到小排序
        reward_sums = history_rewards.sum(axis=0)
        display_indices = sorted(set(display_indices), key=lambda idx: -reward_sums[idx])

        x = np.arange(len(display_indices))
        rewards_selected = history_rewards[:, display_indices]

        fig = ax.figure
        fig.set_size_inches(16, 6)

        # 分为 10 段，每段为 T 的 10%
        segs = []
        seg_size = max(1, T // 10)
        for i in range(9):
            segs.append((i * seg_size, (i + 1) * seg_size))
        segs.append((9 * seg_size, T))

        bottom = np.zeros(len(display_indices))
        cmap = plt.get_cmap('tab10', 10)
        handles = []
        labels = []
        for idx, (start, end) in enumerate(segs):
            seg_rewards = rewards_selected[start:end].sum(axis=0)
            bars = ax.bar(x, seg_rewards, bottom=bottom, color=cmap(idx), width=0.8, label=f'{(idx+1)*10}%', alpha=0.85)
            handles.append(bars[0])
            labels.append(f'{(idx+1)*10}%')
            bottom += seg_rewards

        # 用选中次数标记被选中的曲线
        for i, idx_curve in enumerate(display_indices):
            if idx_curve in chosen_count_dict:
                ax.bar(x[i], bottom[i], width=0.8, color='red', alpha=0.18, zorder=0)
                ax.text(x[i], bottom[i]+0.01, f'*{chosen_count_dict[idx_curve]}', ha='center', va='bottom', fontsize=8, color='red')

        # 按照 10% ~ 100% 分十段
        y_max = bottom.max()
        yticks = [y_max * i / 10 for i in range(1, 11)]
        yticklabels = [f'{i*10}%' for i in range(1, 11)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        # 设置图表属性
        ax.set_title('Stacked Reward per Price Curve')
        ax.set_xlabel('Price Curve Index (Sampled + Chosen)')
        ax.set_ylabel('Total Reward (Stacked by Round)')
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        ax.set_xticks([])
        ax.legend(handles, labels, title='Rounds (by 10%)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    def plot_valuation_functions(self, ax):
        """
        绘制买家估值函数
        """
        for buyer_type in range(self.m):
            vals = [self.valuation(buyer_type, n) for n in self.quantities]
            ax.plot(self.quantities, vals, label=f'Buyer {buyer_type}')
        
        # 设置图表属性
        ax.set_title('Buyer Valuation Functions')
        ax.set_xlabel('Quantity')
        ax.set_ylabel('Valuation')
        ax.legend()
        ax.grid(True)        

if __name__ == "__main__":
    N = 100
    m = 20
    J = 2
    T = 20
    epsilon = 0.08
    theta = 1
    sample_nums = 20000

    def valuation(buyer_type, n):
        V = 0.5 + abs(buyer_type - 10) / T
        a = 0.1 + buyer_type * 0.01
        b = 0.2 + buyer_type * 0.01
        return V - b * math.exp(-a * n)

    # 执行 FTPLAgent 的在线对抗定价
    agent = FTPLAgent(N, m, J, T, epsilon, theta, valuation, sample_nums)
    agent.Online_Adversarial_Pricing()
    records = agent.records

    # 逐个输出被选中的价格曲线
    print("选中的价格曲线：")
    for idx in records['chosen_curve_indices']:
        curve = agent.price_curves[idx]
        rounded_curve = [round(p, 2) for p in curve]
        print(f"Curve {idx}: {rounded_curve}")

    # 输出其他记录
    print("\n")
    print("买家类型序列：", records['buyers'])
    print("每轮购买数量：", records['purchase_nums'])
    print("每轮支付金额：", np.round(records['payments'], 2))
    print("每轮累计收益：", np.round(records['cumulative_revenue'], 2))
    print("每轮的历史奖励总和：", np.round(records['history_rewards_sum'], 2))
    print("遗憾：", np.round(records['regret'], 2))

    # 生成可视化图表
    agent.Visualization()