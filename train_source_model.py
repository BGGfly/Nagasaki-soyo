import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, log_loss
import joblib

# 导入SMOTE
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("错误: imbalanced-learn 库未安装。")
    print("请先运行: pip install imbalanced-learn")
    exit()

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
print(f"数据清洗后，原始样本数: {df_clean.shape[0]}。")
print("原始数据集中各类别样本数量:")
print(y.value_counts())


# --- 2. 使用SMOTE进行过采样，平衡数据集 ---
print("\n步骤2: 使用SMOTE对稀有类别进行过采样...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\n过采样后，数据集中各类别样本数量:")
print(y_resampled.value_counts()) # 您会看到所有类别的样本数都变得一样多

# --- 3. 在平衡后的数据集上进行划分 ---
print("\n步骤3: 在平衡后的数据集上划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)
print(f"数据集划分完毕。训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 验证一下测试集中各类别数量
print("\n划分后，测试集中的各类别样本数量:")
print(y_test.value_counts())

# --- 4. 特征标准化 ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n步骤4: 特征标准化完成。")

# --- 5. 训练MLP模型 ---
print("\n步骤5: 开始训练MLP神经网络模型...")
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=300,
    random_state=42,
    early_stopping=True,
    verbose=False
)
model.fit(X_train_scaled, y_train)
print("模型训练完成。")

# --- 6. 保存模型 ---
model_filename = 'trained_source_model_smote.joblib'
joblib.dump(model, model_filename)
print(f"\n步骤6: 训练好的模型已保存至 '{model_filename}'")


# --- 7. 在测试集上评估模型 ---
print("\n步骤7: 开始在测试集上评估模型性能...")
y_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型在源域【测试集】上的最终准确率: {test_accuracy * 100:.2f}%")

fault_labels = {0: '正常 (N)', 1: '外圈 (OR)', 2: '内圈 (IR)', 3: '滚动体 (B)'}
class_names = [fault_labels[i] for i in sorted(y_resampled.unique())]
report_text = classification_report(y_test, y_pred, target_names=class_names)
print("\n详细分类报告:")
print(report_text)

# 保存详细分类报告
report_filename = 'classification_report_smote.txt'
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write(f"模型(SMOTE)在源域测试集上的最终性能报告\n")
    f.write("="*50 + "\n")
    f.write(f"总体准确率 (Accuracy): {test_accuracy * 100:.2f}%\n\n")
    f.write("详细分类报告:\n")
    f.write(report_text)
print(f"\n详细分类报告已保存至 '{report_filename}'")

# --- 8. 输出并保存可视化图像 ---
# (这部分代码与之前版本基本一致，仅修改了输出文件名以作区分)
print("\n步骤8: 开始生成并保存可视化图像...")
# ... (此处省略重复的绘图代码，实际运行时请保留)
# 训练过程曲线
plt.figure() # 创建新图形，避免重叠
fig, ax1 = plt.subplots(figsize=(12, 5))
color = 'tab:red'
ax1.set_xlabel('训练轮次 (Epoch)')
ax1.set_ylabel('训练损失 (Loss)', color=color)
ax1.plot(model.loss_curve_, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('验证准确率 (Validation Accuracy)', color=color)
ax2.plot(model.validation_scores_, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.suptitle('模型训练过程指标 (SMOTE)', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_history_smote.png')
print("  - 训练过程曲线图已保存至 'training_history_smote.png'")

# 测试集混淆矩阵
plt.figure()
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, display_labels=class_names, cmap='Blues')
ax.set_title('测试集混淆矩阵 (SMOTE)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('test_confusion_matrix_smote.png')
print("  - 测试集混淆矩阵图已保存至 'test_confusion_matrix_smote.png'")

# 测试集详细性能指标条形图
plt.figure()
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
metrics_df = report_df[['precision', 'recall', 'f1-score']].iloc[:-3]
fig, ax = plt.subplots(figsize=(12, 7))
metrics_df.plot(kind='bar', ax=ax)
ax.set_title('测试集上各故障类别的性能指标对比 (SMOTE)', fontsize=16)
ax.set_xlabel('故障类型', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_xticklabels(metrics_df.index, rotation=45, ha="right")
ax.legend(title='指标')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('test_set_metrics_barchart_smote.png')
print("  - 测试集性能指标条形图已保存至 'test_set_metrics_barchart_smote.png'")

print("\n所有任务完成。")