"""
stepX_preprocess_target.py
目标域数据处理脚本
 - 小波去噪
 - 信号级 Z-score 标准化
 - 滑窗分片 + 特征提取（13维）
 - 频率归一化（按目标转速保持或对齐源域）
 - 特征级标准化（使用源域均值/标准差）
 - 输出: features_target_ready.npy
"""

import os
import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from tqdm import tqdm
import pandas as pd
from scipy.io import loadmat

# -------------------- 小波去噪 --------------------
def wavelet_denoise(signal):
    wavelet, level = 'db8', 5
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    return pywt.waverec(new_coeffs, wavelet)[:len(signal)]

# -------------------- 频率特征归一化 --------------------
def rpm_normalize_features(X, freq_cols, rpm_src, rpm_target=600):
    X_norm = X.copy()
    scale = rpm_target / rpm_src
    X_norm[:, freq_cols] = X[:, freq_cols] * scale
    return X_norm

# -------------------- 特征级标准化 --------------------
def feature_level_standardize(X, mean, std):
    std[std == 0] = 1.0  # 防止除以0
    X_std = (X - mean) / std
    return X_std

# -------------------- 计算小波熵 --------------------
def calculate_entropy(coeffs):
    coeffs = [c for c in coeffs if c.any()]
    if not coeffs:
        return 0
    energy = [np.sum(c**2) for c in coeffs]
    total = np.sum(energy)
    if total == 0:
        return 0
    p = energy / total
    return -np.sum(p * np.log2(p + 1e-6))

# -------------------- 特征提取 --------------------
def extract_features(x, fs):
    feats = {}
    # 时域
    rms = np.sqrt(np.mean(x**2))
    peak, mean_abs = np.max(np.abs(x)), np.mean(np.abs(x))
    feats['time_rms'] = rms
    feats['time_kurtosis'] = kurtosis(x)
    feats['time_skewness'] = skew(x)
    feats['time_crest_factor'] = peak / (rms + 1e-6)
    feats['time_shape_factor'] = rms / (mean_abs + 1e-6)
    feats['time_impulse_factor'] = peak / (mean_abs + 1e-6)
    # 频域
    N = len(x)
    yf, xf = fft(x), fftfreq(N, 1/fs)
    half = N // 2
    mag, freqs = np.abs(yf[:half]), xf[:half]
    sum_mag = np.sum(mag) + 1e-6
    feats['freq_centroid'] = np.sum(freqs * mag) / sum_mag
    feats['freq_rms'] = np.sqrt(np.sum(freqs**2 * mag) / sum_mag)
    feats['freq_variance'] = np.sum(((freqs - feats['freq_centroid'])**2) * mag) / sum_mag
    env = np.abs(hilbert(x))
    env_fft = np.abs(fft(env - np.mean(env)))[:half]
    peak_idx = np.argmax(env_fft[1:]) + 1
    feats['freq_envelope_peak_freq'] = freqs[peak_idx]
    # 时频
    wp = pywt.WaveletPacket(data=x, wavelet='db8', maxlevel=3)
    nodes = wp.get_level(3, order='natural')
    coeffs = [n.data for n in nodes]
    feats['wp_entropy_low'] = calculate_entropy(coeffs[:2])
    feats['wp_entropy_mid'] = calculate_entropy(coeffs[2:5])
    feats['wp_entropy_high'] = calculate_entropy(coeffs[5:])
    return list(feats.values())

# -------------------- 主流程 --------------------
def main():
    DATA_ROOT = os.path.join('data', 'target')  # 目标域数据目录
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"未找到目标域数据目录: {DATA_ROOT}")

    all_files = [os.path.join(DATA_ROOT, f) for f in os.listdir(DATA_ROOT) if f.endswith('.mat')]
    print(f"检测到 {len(all_files)} 个目标域 .mat 文件")

    fs, win, step = 12000, 2048, 1024
    freq_cols = [6,7,8,9]  # 频率特征列索引

    # 加载源域特征的 mean/std，用于标准化目标域特征
    scaler = np.load('feature_scaler.npz')
    mean_src, std_src = scaler['mean'], scaler['std']

    X_target = []

    for file_path in tqdm(all_files, desc="处理目标域文件"):
        try:
            data_dict = loadmat(file_path)
            key = next(k for k in data_dict if k[0].isalpha())  # 取唯一振动信号变量
            signal = data_dict[key].flatten()
        except Exception as e:
            print(f"⚠️ 跳过文件 {file_path}: {e}")
            continue

        # 小波去噪
        signal = wavelet_denoise(signal)
        # 信号级 Z-score 标准化
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        # 滑窗分片 + 特征提取
        for start in range(0, len(signal) - win + 1, step):
            sl = signal[start:start+win]
            feats = extract_features(sl, fs)
            X_target.append(feats)

    X_target = np.array(X_target)

    # 频率归一化（目标域保持 600 rpm，可选缩放）
    rpm_target = 600
    rpm_src = rpm_target  # 因为目标域就是标准转速，可保持原值
    X_target = rpm_normalize_features(X_target, freq_cols, rpm_src, rpm_target)

    # 特征级标准化（使用源域 mean/std）
    X_target_std = feature_level_standardize(X_target, mean_src, std_src)

    # 保存目标域特征矩阵
    np.save('features_target_ready.npy', X_target_std)
    print(f"✅ 目标域处理完成，特征矩阵形状: {X_target_std.shape}")

if __name__ == "__main__":
    main()
