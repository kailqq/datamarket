import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 全局变量存储文件夹名称
OUTPUT_FOLDER = None

def create_output_folder():
    global OUTPUT_FOLDER
    folder_name = f"UCB_visualization"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    OUTPUT_FOLDER = folder_name
    return folder_name


def visualize_price_curves(price_curves, N):
    """
    Visualize price curves collection
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    labels = ['Free Price', 'Linear Low Price', 'Linear High Price', 'Fixed Low Price', 'Fixed High Price']
    
    for i, curve in enumerate(price_curves):
        plt.plot(range(N+1), curve, 
                color=colors[i % len(colors)], 
                linewidth=2, 
                label=labels[i] if i < len(labels) else f'Curve {i}',
                marker='o' if i == 0 else None,
                markersize=3)
    
    plt.xlabel('Number of Data Points (n)', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title('Price Curves Collection', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save instead of show
    if OUTPUT_FOLDER:
        plt.savefig(f'{OUTPUT_FOLDER}/01_price_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_valuation_functions(valuation_function, N, m_types=3):
    """
    Visualize buyer valuation functions
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: All types' valuation functions
    n_range = np.arange(0, N+1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, m_types))
    type_names = [f'Type {i}' for i in range(m_types)]
    
    for buyer_type in range(m_types):
        values = [valuation_function(buyer_type, n, N) for n in n_range]
        axes[0].plot(n_range, values, 
                    color=colors[buyer_type], 
                    linewidth=2, 
                    label=type_names[buyer_type],
                    marker='o',
                    markersize=4)
    
    axes[0].set_xlabel('Number of Data Points (n)', fontsize=12)
    axes[0].set_ylabel('Valuation', fontsize=12)
    axes[0].set_title('Buyer Valuation Functions', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Marginal utility (diminishing returns)
    for buyer_type in range(m_types):
        values = [valuation_function(buyer_type, n, N) for n in n_range]
        marginal_values = [0] + [values[i] - values[i-1] for i in range(1, len(values))]
        axes[1].plot(n_range, marginal_values, 
                    color=colors[buyer_type], 
                    linewidth=2, 
                    label=type_names[buyer_type],
                    marker='s',
                    markersize=4)
    
    axes[1].set_xlabel('Number of Data Points (n)', fontsize=12)
    axes[1].set_ylabel('Marginal Utility', fontsize=12)
    axes[1].set_title('Marginal Utility (Diminishing Returns)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save instead of show
    if OUTPUT_FOLDER:
        plt.savefig(f'{OUTPUT_FOLDER}/02_valuation_functions.png', dpi=300, bbox_inches='tight')
    plt.close()



def visualize_learning_process(agent_history, m_types, num_curves, Q_TRUE=None):
    """
    Visualize UCB learning process
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cumulative revenue
    rounds = agent_history['round']
    cumulative_revenue = agent_history['cumulative_revenue']
    axes[0, 0].plot(rounds, cumulative_revenue, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Round Number', fontsize=12)
    axes[0, 0].set_ylabel('Cumulative Revenue', fontsize=12)
    axes[0, 0].set_title('Cumulative Revenue Change', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Type distribution estimation
    if len(agent_history['T_i_history']) > 0:
        T_i_history = np.array(agent_history['T_i_history'])
        N_i_history = np.array(agent_history['N_i_history'])
        
        # Avoid division by zero
        q_estimates_history = np.divide(N_i_history, np.maximum(T_i_history, 1))
        
        # 动态生成颜色和名称
        colors = plt.cm.tab10(np.linspace(0, 1, m_types))
        type_names = [f'Type {i}' for i in range(m_types)]
        
        for i in range(m_types):
            axes[0, 1].plot(rounds, q_estimates_history[:, i], 
                           color=colors[i], linewidth=2, label=type_names[i])
            if Q_TRUE is not None:
                axes[0, 1].axhline(y=Q_TRUE[i], color=colors[i], linestyle='--', alpha=0.7)
        
        axes[0, 1].set_xlabel('Round Number', fontsize=12)
        axes[0, 1].set_ylabel('Type Distribution Estimate', fontsize=12)
        axes[0, 1].set_title('Type Distribution Learning Process', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if Q_TRUE is not None:
            axes[0, 1].text(0.02, 0.98, 'Dashed line: True value', transform=axes[0, 1].transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Action selection distribution
    actions = agent_history['action']
    action_counts = np.bincount(actions, minlength=num_curves)
    action_labels = [f'Curve {i}' for i in range(num_curves)]
    
    # 动态生成颜色
    bar_colors = plt.cm.Set3(np.linspace(0, 1, num_curves))
    
    bars = axes[1, 0].bar(range(num_curves), action_counts, color=bar_colors)
    axes[1, 0].set_xlabel('Price Curve Index', fontsize=12)
    axes[1, 0].set_ylabel('Selection Count', fontsize=12)
    axes[1, 0].set_title('Action Selection Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(num_curves))
    axes[1, 0].set_xticklabels(action_labels)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(action_counts),
                       f'{int(height)}', ha='center', va='bottom')
    
    # 4. Exploration vs Exploitation
    window_size = max(100, len(rounds) // 20)
    if len(rounds) > window_size:
        exploration_ratio = []
        for i in range(window_size, len(rounds)):
            recent_actions = actions[i-window_size:i]
            unique_actions = len(set(recent_actions))
            exploration_ratio.append(unique_actions / num_curves)
        
        axes[1, 1].plot(rounds[window_size:], exploration_ratio, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Round Number', fontsize=12)
        axes[1, 1].set_ylabel('Exploration Ratio', fontsize=12)
        axes[1, 1].set_title(f'Exploration vs Exploitation (Window Size: {window_size})', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save instead of show
    if OUTPUT_FOLDER:
        plt.savefig(f'{OUTPUT_FOLDER}/04_learning_process.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_confidence_bounds(agent_history, m_types):
    """
    Visualize confidence bounds
    """
    if len(agent_history['q_estimates']) == 0:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    rounds = agent_history['round']
    q_estimates = np.array(agent_history['q_estimates'])
    confidence_bounds = np.array(agent_history['confidence_bounds'])
    
    # 动态生成颜色和名称
    colors = plt.cm.tab10(np.linspace(0, 1, m_types))
    type_names = [f'Type {i}' for i in range(m_types)]
    
    # Left plot: Confidence intervals
    for i in range(m_types):
        lower_bound = q_estimates[:, i] - confidence_bounds[:, i]
        upper_bound = q_estimates[:, i] + confidence_bounds[:, i]
        
        axes[0].plot(rounds, q_estimates[:, i], color=colors[i], linewidth=2, label=type_names[i])
        axes[0].fill_between(rounds, lower_bound, upper_bound, 
                           color=colors[i], alpha=0.3)
    
    axes[0].set_xlabel('Round Number', fontsize=12)
    axes[0].set_ylabel('Type Distribution Estimate', fontsize=12)
    axes[0].set_title('Confidence Intervals', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Confidence width
    for i in range(m_types):
        axes[1].plot(rounds, confidence_bounds[:, i], 
                    color=colors[i], linewidth=2, label=type_names[i])
    
    axes[1].set_xlabel('Round Number', fontsize=12)
    axes[1].set_ylabel('Confidence Width', fontsize=12)
    axes[1].set_title('Confidence Width Change', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save instead of show
    if OUTPUT_FOLDER:
        plt.savefig(f'{OUTPUT_FOLDER}/05_confidence_bounds.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_visualization_suite(M_TYPES, N_ITEMS, T_HORIZON, Q_TRUE, price_curves, 
                          valuation_function, revenues, purchases, agent):
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    print(f"图片将保存到文件夹: {output_folder}")
    
    # 1. 生成和可视化价格曲线
    print("\n1. 生成价格曲线集合...")
    print(f"生成了 {len(price_curves)} 条价格曲线")
    visualize_price_curves(price_curves, N_ITEMS)
    print("✅ 价格曲线图已保存: 01_price_curves.png")
    
    # 2. 可视化估值函数
    print("\n2. 可视化买家估值函数...")
    visualize_valuation_functions(valuation_function, N_ITEMS, M_TYPES)
    print("✅ 估值函数图已保存: 02_valuation_functions.png")
    
    # 3. 可视化学习过程
    print("\n3. 可视化学习结果...")
    visualize_learning_process(agent.history, M_TYPES, agent.num_curves, Q_TRUE)
    print("✅ 学习过程图已保存: 04_learning_process.png")
    
    visualize_confidence_bounds(agent.history, M_TYPES)
    print("✅ 置信度区间图已保存: 05_confidence_bounds.png")
 
    print(f"\n所有图片已保存到文件夹: {output_folder}")
    
    return output_folder
