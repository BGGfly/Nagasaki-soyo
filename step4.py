"""
Step3.5 + Step4: 源域频率归一化 + 特征级标准化
 - 输入: step3_features.npy, step3_labels.npy
 - 输出: features_ready.npy, labels_ready.npy
 - 保存标准化参数: feature_scaler.npz
"""

import numpy as np

# -------------------- 频率归一化 --------------------
def rpm_normalize_features(X, freq_cols, rpm_src, rpm_target=600):
    """
    将源域频率特征归一化到目标域转速
    """
    X_norm = X.copy()
    scale = rpm_target / rpm_src
    X_norm[:, freq_cols] = X[:, freq_cols] * scale
    return X_norm

# -------------------- 特征级标准化 --------------------
def feature_level_standardize(X, mean=None, std=None):
    """
    对特征矩阵做 Z-score 标准化
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0  # 防止除以0
    X_std = (X - mean) / std
    return X_std, mean, std

# -------------------- 主流程 --------------------
def main():
    # 加载 Step3 特征和标签
    X = np.load('step3_features.npy')
    y = np.load('step3_labels.npy')

    # -------------------- 频率归一化 --------------------
    freq_cols = [6,7,8,9]        # freq_centroid, freq_rms, freq_variance, freq_envelope_peak_freq
    rpm_src = 1797               # 源域转速
    rpm_target = 600             # 目标域转速
    X_norm = rpm_normalize_features(X, freq_cols, rpm_src, rpm_target)
    print(f"✅ 频率特征已归一化到 {rpm_target} rpm")

    # -------------------- 特征级标准化 --------------------
    X_std, mean, std = feature_level_standardize(X_norm)
    np.save('features_ready.npy', X_std)
    np.save('labels_ready.npy', y)
    np.savez('feature_scaler.npz', mean=mean, std=std)
    print(f"✅ 特征级标准化完成，特征矩阵可直接用于迁移学习")
    print(f"特征矩阵形状: {X_std.shape}, 标签形状: {y.shape}")

if __name__ == "__main__":
    main()
