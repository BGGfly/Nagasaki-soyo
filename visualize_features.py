import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略一些不影响结果的警告
warnings.filterwarnings('ignore')

# --- 1. 设置：加载数据并配置画图环境 ---

# --- 【重要】配置字体以支持中文显示 ---
# 这一步是为了确保图表中的中文标签能正确显示，而不是方框。
# 我们尝试寻找系统中常见的几种中文字体。
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Heiti TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception as e:
    print(f"中文字体设置失败: {e}")
    print("警告：图表中的中文可能无法正常显示。")

# 加载特征数据文件
try:
    df = pd.read_csv('source_data_sliced_features.csv')
    print("成功加载 'source_data_sliced_features.csv'")
    print(f"数据集包含 {df.shape[0]} 个样本和 {df.shape[1]} 个字段。")
except FileNotFoundError:
    print("错误: 'source_data_sliced_features.csv' 文件未找到。请先运行特征提取脚本。")
    exit()

# --- 2. 关键特征的箱线图可视化 ---
print("\n正在生成关键特征的箱线图...")

# 挑选几个有代表性的特征进行可视化
features_to_plot = [
    'time_rms',  # 时域均方根值 (反映信号能量)
    'time_kurtosis',  # 时域峭度 (反映信号的冲击性)
    'time_crest_factor',  # 时域峰值因子 (反映信号的峰值水平)
    'freq_centroid'  # 频域质心 (反映频率分量的中心位置)
]

# 设置图表风格
sns.set(style="whitegrid")

for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    # 使用 seaborn 绘制箱线图，并指定类别顺序
    sns.boxplot(x='FaultType_Str', y=feature, data=df, order=['N', 'B', 'IR', 'OR'])
    plt.title(f'不同故障类型下的"{feature}"特征分布', fontsize=16)
    plt.xlabel('故障类型', fontsize=12)
    plt.ylabel(f'特征值 ({feature})', fontsize=12)

    # 将图表保存为文件
    output_filename = f"boxplot_{feature}.png"
    plt.savefig(output_filename)
    print(f"已将箱线图保存至 {output_filename}")
    # 如果您在Jupyter Notebook等交互式环境中运行，可以使用 plt.show() 来直接显示图表

print("\n箱线图生成完毕。")

# --- 3. 整体特征空间的t-SNE降维可视化 ---
print("\n正在进行t-SNE降维可视化... (这一步可能需要几分钟)")

# 挑选出所有的特征列
feature_columns = [
    'time_mean', 'time_std', 'time_rms', 'time_peak', 'time_p2p',
    'time_kurtosis', 'time_skewness', 'time_crest_factor', 'time_shape_factor',
    'time_impulse_factor', 'freq_centroid', 'freq_rms', 'freq_variance'
]

# t-SNE无法处理NaN或无穷大的值，需要先进行清洗
df_clean = df.dropna(subset=feature_columns).copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)

if df_clean.shape[0] < 50:
    print("警告：数据清洗后样本过少，无法进行t-SNE分析。")
else:
    # 分离特征和标签
    features = df_clean[feature_columns]
    labels = df_clean['FaultType_Str']

    # 特征标准化 (对于t-SNE非常重要)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 初始化并运行t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    features_2d = tsne.fit_transform(features_scaled)

    # 创建一个新的DataFrame用于绘图
    tsne_df = pd.DataFrame(data=features_2d, columns=['维度1', '维度2'])
    tsne_df['故障类型'] = labels.values

    # 绘制t-SNE结果散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="维度1", y="维度2",
        hue="故障类型",
        hue_order=['N', 'B', 'IR', 'OR'],  # 指定图例顺序
        palette=sns.color_palette("hsv", 4),  # 使用鲜艳的调色板
        data=tsne_df,
        legend="full",
        alpha=0.7
    )
    plt.title('轴承故障特征的t-SNE可视化', fontsize=16)
    plt.xlabel('t-SNE 维度1', fontsize=12)
    plt.ylabel('t-SNE 维度2', fontsize=12)

    # 保存图表
    output_filename_tsne = "tsne_visualization.png"
    plt.savefig(output_filename_tsne)
    print(f"已将t-SNE图保存至 {output_filename_tsne}")

    print("\nt-SNE可视化生成完毕。")