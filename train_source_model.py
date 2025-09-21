import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier # 替换为MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, log_loss

# --- 设置字体以支持中文显示 ---
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Heiti TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字体设置失败: {e}")

# --- 1. 加载数据并准备 ---
try:
    df = pd.read_csv('source_data_sliced_features.csv')
    print("步骤1: 成功加载特征文件 'source_data_sliced_features.csv'")
except FileNotFoundError:
    print("错误: 'source_data_sliced_features.csv' 文件未找到。")
    exit()

feature_columns = [
    'time_mean', 'time_std', 'time_rms', 'time_peak', 'time_p2p',
    'time_kurtosis', 'time_skewness', 'time_crest_factor', 'time_shape_factor',
    'time_impulse_factor', 'freq_centroid', 'freq_rms', 'freq_variance'
]
target_column = 'FaultType_Label'

df_clean = df.dropna(subset=feature_columns + [target_column])
X = df_clean[feature_columns]
y = df_clean[target_column]
print(f"数据清洗后，用于训练的总样本数: {df_clean.shape[0]}。")

# --- 2. 划分训练集和测试集 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"步骤2: 数据集划分完毕。训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# --- 3. 特征标准化 ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("步骤3: 特征标准化完成。")

# --- 4. 训练MLP模型 ---
print("\n步骤4: 开始训练MLP神经网络模型...")
# 初始化MLP分类器
# hidden_layer_sizes: 定义了两个隐藏层，分别有100和50个神经元
# max_iter: 最大训练轮次 (epochs)
# early_stopping=True: 当验证分数不再提升时，提前终止训练，防止过拟合
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=300,
    random_state=42,
    early_stopping=True,
    verbose=False # 设置为True可以看到每轮的loss
)

model.fit(X_train_scaled, y_train)
print("模型训练完成。")

# --- 5. 性能评估 ---
print("\n步骤5: 开始在测试集上评估模型性能...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# 计算测试集上的指标
test_accuracy = accuracy_score(y_test, y_pred)
test_loss = log_loss(y_test, y_pred_proba) # 使用log_loss计算测试集损失

print(f"\n模型在源域【测试集】上的最终指标:")
print(f"  - 最终准确率 (Accuracy): {test_accuracy * 100:.2f}%")
print(f"  - 最终损失 (Log Loss): {test_loss:.4f}")

fault_labels = {0: '正常 (N)', 1: '外圈 (OR)', 2: '内圈 (IR)', 3: '滚动体 (B)'}
class_names = [fault_labels[i] for i in sorted(y.unique())]
report = classification_report(y_test, y_pred, target_names=class_names)
print("\n详细分类报告:")
print(report)

# --- 6. 输出并保存在训练过程中的可视化图像 ---
print("\n步骤6: 开始生成并保存可视化图像...")

# 6.1 绘制训练过程中的损失和验证准确率曲线
fig, ax1 = plt.subplots(figsize=(12, 5))

# 绘制训练损失曲线
color = 'tab:red'
ax1.set_xlabel('训练轮次 (Epoch)')
ax1.set_ylabel('训练损失 (Loss)', color=color)
ax1.plot(model.loss_curve_, color=color, label='训练损失')
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个Y轴，共享X轴，用于绘制验证准确率
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('验证准确率 (Validation Accuracy)', color=color)
ax2.plot(model.validation_scores_, color=color, label='验证准确率')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('模型训练过程指标', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_history.png')
print("  - 训练过程曲线图已保存至 'training_history.png'")

# 6.2 绘制测试集上的混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    ax=ax,
    display_labels=class_names,
    cmap='Blues'
)
ax.set_title('测试集混淆矩阵')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('test_confusion_matrix.png')
print("  - 测试集混淆矩阵图已保存至 'test_confusion_matrix.png'")

print("\n所有可视化图像均已保存在当前工作目录下。")