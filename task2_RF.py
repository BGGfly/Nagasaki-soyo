import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------- 参数 --------------------
FEATURE_FILE = 'step3_features.npy'
LABEL_FILE = 'step3_labels.npy'
OUTPUT_DIR = 'figures/task2'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -------------------- 1. 加载数据 --------------------
X = np.load(FEATURE_FILE)
y = np.load(LABEL_FILE)
label_names = ['Normal','OR','IR','B']

# -------------------- 2. 划分训练集和测试集 --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------- 3. 特征标准化 --------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- 4. 随机森林训练 --------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# -------------------- 5. 模型评价 --------------------
acc = accuracy_score(y_test, y_pred)
print(f"🔹 测试集准确率: {acc*100:.2f}%\n")
print("🔹 分类报告:")
print(classification_report(y_test, y_pred, target_names=label_names))

# -------------------- 6. 混淆矩阵 --------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# -------------------- 7. 每类预测准确率柱状图 --------------------
class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(6,4))
sns.barplot(x=label_names, y=class_acc*100, palette='Set2')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, v in enumerate(class_acc*100):
    plt.text(i, v+1, f'{v*100:.1f}%', ha='center')
plt.title('Per-Class Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_accuracy.png'))
plt.close()

# -------------------- 8. 特征重要性柱状图 --------------------
feat_importances = clf.feature_importances_
feature_names = [
    'time_rms', 'time_kurtosis', 'time_skewness', 'time_crest_factor',
    'time_shape_factor', 'time_impulse_factor',
    'freq_centroid', 'freq_rms', 'freq_variance', 'freq_envelope_peak_freq',
    'wp_entropy_low', 'wp_entropy_mid', 'wp_entropy_high'
]
plt.figure(figsize=(10,5))
sns.barplot(x=feature_names, y=feat_importances, palette='coolwarm')
plt.xticks(rotation=45)
plt.ylabel('Importance')
plt.title('Feature Importance - RandomForest')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
plt.close()

# -------------------- 9. PCA 散点图 --------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
plt.figure(figsize=(6,6))
for label in np.unique(y_train):
    plt.scatter(X_pca[y_train==label,0], X_pca[y_train==label,1], label=label_names[label], alpha=0.6)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA - Training Samples')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_train.png'))
plt.close()

# -------------------- 10. t-SNE 散点图 --------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_train_scaled)
plt.figure(figsize=(6,6))
for label in np.unique(y_train):
    plt.scatter(X_tsne[y_train==label,0], X_tsne[y_train==label,1], label=label_names[label], alpha=0.6)
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('t-SNE - Training Samples')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_train.png'))
plt.close()

print(f"✅ 源域诊断可视化完成，所有图片保存到 {OUTPUT_DIR}/")
