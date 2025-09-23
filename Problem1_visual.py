import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------- 参数 --------------------
FEATURE_FILE = 'step3_features.npy'   # 源域特征文件
LABEL_FILE = 'step3_labels.npy'       # 源域标签文件
OUTPUT_DIR = os.path.join('figures', 'source')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -------------------- 加载数据 --------------------
X = np.load(FEATURE_FILE)
y = np.load(LABEL_FILE)

feature_names = [
    'time_rms', 'time_kurtosis', 'time_skewness', 'time_crest_factor',
    'time_shape_factor', 'time_impulse_factor',
    'freq_centroid', 'freq_rms', 'freq_variance', 'freq_envelope_peak_freq',
    'wp_entropy_low', 'wp_entropy_mid', 'wp_entropy_high'
]

# 故障类型映射
label_map = {0: 'Normal', 1: 'OR', 2: 'IR', 3: 'B'}
unique, counts = np.unique(y, return_counts=True)
print("🔹 源域各类别样本数：")
for u, c in zip(unique, counts):
    print(f"  {label_map[u]}: {c} 个样本")

# -------------------- 1. 单特征直方图 --------------------
for i, name in enumerate(feature_names):
    plt.figure(figsize=(6,4))
    plt.hist(X[:, i], bins=50, color='skyblue', edgecolor='black')
    plt.title(f'{name} Distribution')
    plt.xlabel(name)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_hist.png'))
    plt.close()

# -------------------- 2. 特征箱线图 --------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=X)
plt.xticks(ticks=range(len(feature_names)), labels=feature_names, rotation=45)
plt.title('Feature Boxplot')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_boxplot.png'))
plt.close()

# -------------------- 3. 特征相关性热图 --------------------
corr = np.corrcoef(X.T)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', xticklabels=feature_names, yticklabels=feature_names, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation.png'))
plt.close()

# -------------------- 4. PCA 降维散点图 --------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(6,6))
for u in unique:
    plt.scatter(X_pca[y==u,0], X_pca[y==u,1], label=label_map[u], alpha=0.6)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Scatter Plot')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_scatter.png'))
plt.close()

# -------------------- 5. t-SNE 降维散点图 --------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(6,6))
for u in unique:
    plt.scatter(X_tsne[y==u,0], X_tsne[y==u,1], label=label_map[u], alpha=0.6)
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('t-SNE Scatter Plot')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_scatter.png'))
plt.close()

plt.figure(figsize=(6,4))
sns.barplot(x=[label_map[u] for u in unique], y=counts, palette="Set2")
plt.xlabel('Fault Type')
plt.ylabel('Number of Samples')
plt.title('Source Domain Sample Count per Class')
for i, c in enumerate(counts):
    plt.text(i, c + 2, str(c), ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_count.png'))
plt.close()
print(f"✅ 源域特征可视化完成，所有图片已保存到 {OUTPUT_DIR}/")
