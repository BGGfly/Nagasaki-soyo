import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

# -------------------- 参数 --------------------
SOURCE_FEATURE_FILE = 'step3_features.npy'
SOURCE_LABEL_FILE   = 'step3_labels.npy'
CWRU_FEATURE_FILE   = 'stepX_CWRU_features.npy'
CWRU_LABEL_FILE     = 'stepX_CWRU_labels.npy'
OUTPUT_DIR = os.path.join('figures', 'task3')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -------------------- 加载数据 --------------------
X_source = np.load(SOURCE_FEATURE_FILE)
y_source = np.load(SOURCE_LABEL_FILE)
X_cwru = np.load(CWRU_FEATURE_FILE)
y_cwru = np.load(CWRU_LABEL_FILE)

feature_names = [
    'time_rms', 'time_kurtosis', 'time_skewness', 'time_crest_factor',
    'time_shape_factor', 'time_impulse_factor',
    'freq_centroid', 'freq_rms', 'freq_variance', 'freq_envelope_peak_freq',
    'wp_entropy_low', 'wp_entropy_mid', 'wp_entropy_high'
]

# -------------------- 划分 CWRU 微调集 / 测试集 --------------------
X_cwru_train, X_cwru_test, y_cwru_train, y_cwru_test = train_test_split(
    X_cwru, y_cwru, test_size=0.995, random_state=42, stratify=y_cwru
)

# -------------------- CORAL 域自适应 --------------------
def coral(source, target):
    source_c = source - np.mean(source, axis=0)
    target_c = target - np.mean(target, axis=0)
    cov_s = np.cov(source_c, rowvar=False) + np.eye(source.shape[1])
    cov_t = np.cov(target_c, rowvar=False) + np.eye(target.shape[1])
    U_s, S_s, _ = np.linalg.svd(cov_s)
    U_t, S_t, _ = np.linalg.svd(cov_t)
    A_coral = U_s @ np.diag(1.0/np.sqrt(S_s)) @ U_s.T
    B_coral = U_t @ np.diag(np.sqrt(S_t)) @ U_t.T
    source_aligned = (source_c @ A_coral) @ B_coral
    source_aligned += np.mean(target, axis=0)
    return source_aligned

X_source_aligned = coral(X_source, X_cwru_train)

# -------------------- 源域 + 半监督微调 --------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_source_aligned, y_source)
clf.fit(X_cwru_train, y_cwru_train)

# -------------------- 测试集预测与性能 --------------------
y_pred = clf.predict(X_cwru_test)
report = classification_report(y_cwru_test, y_pred, digits=4)
print("CWRU 测试集分类报告:\n", report)

# -------------------- 保存预测标签 --------------------
np.save(os.path.join(OUTPUT_DIR, 'cwru_test_pred_labels.npy'), y_pred)

# -------------------- 可视化函数 --------------------
def visualize_all(X_s, y_s, X_cwru_train, y_cwru_train, X_cwru_test, y_cwru_test, y_cwru_pred, output_dir):
    # --- 1. 单特征直方图 ---
    all_X = [X_s, X_cwru_train, X_cwru_test]
    all_labels = ['Source', 'CWRU_train', 'CWRU_test']
    colors = ['blue', 'green', 'red']
    for i, name in enumerate(feature_names):
        plt.figure(figsize=(8,5))
        for X_part, label, color in zip(all_X, all_labels, colors):
            plt.hist(X_part[:, i], bins=50, alpha=0.5, label=label, color=color)
        plt.title(f'{name} Distribution')
        plt.xlabel(name)
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hist_{name}.png'))
        plt.close()

    # --- 2. 箱线图 ---
    plt.figure(figsize=(14,6))
    sns.boxplot(data=np.vstack(all_X))
    plt.xticks(ticks=range(len(feature_names)), labels=feature_names, rotation=45)
    plt.title('Boxplot All Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_all_features.png'))
    plt.close()

    # --- 3. 相关性热图 ---
    corr = np.corrcoef(np.vstack(all_X).T)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', xticklabels=feature_names, yticklabels=feature_names, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
    plt.close()

    # --- 4. PCA ---
    X_combined = np.vstack([X_s, X_cwru_train, X_cwru_test])
    labels_combined = ['Source']*len(X_s) + ['CWRU_train']*len(X_cwru_train) + ['CWRU_test']*len(X_cwru_test)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:len(X_s),0], X_pca[:len(X_s),1], c='blue', label='Source', alpha=0.6)
    plt.scatter(X_pca[len(X_s):len(X_s)+len(X_cwru_train),0],
                X_pca[len(X_s):len(X_s)+len(X_cwru_train),1],
                c='green', label='CWRU_train', alpha=0.6)
    plt.scatter(X_pca[len(X_s)+len(X_cwru_train):,0],
                X_pca[len(X_s)+len(X_cwru_train):,1],
                c='red', label='CWRU_test', alpha=0.6)
    plt.title('PCA Visualization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_all.png'))
    plt.close()

    # --- 5. t-SNE ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_combined)
    plt.figure(figsize=(6,6))
    plt.scatter(X_tsne[:len(X_s),0], X_tsne[:len(X_s),1], c='blue', label='Source', alpha=0.6)
    plt.scatter(X_tsne[len(X_s):len(X_s)+len(X_cwru_train),0],
                X_tsne[len(X_s):len(X_s)+len(X_cwru_train),1],
                c='green', label='CWRU_train', alpha=0.6)
    plt.scatter(X_tsne[len(X_s)+len(X_cwru_train):,0],
                X_tsne[len(X_s)+len(X_cwru_train):,1],
                c='red', label='CWRU_test', alpha=0.6)
    plt.title('t-SNE Visualization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_all.png'))
    plt.close()

    # --- 6. 混淆矩阵 ---
    cm = confusion_matrix(y_cwru_test, y_cwru_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (CWRU Test)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

visualize_all(X_source_aligned, y_source, X_cwru_train, y_cwru_train,
              X_cwru_test, y_cwru_test, y_pred, OUTPUT_DIR)

print(f"✅ 可视化完成，所有图片已保存到 {OUTPUT_DIR}/")
joblib.dump(clf, 'task3_source_model.pkl')
print("✅ 源域模型已保存为 task3_source_model.pkl")
