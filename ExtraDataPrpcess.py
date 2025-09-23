"""
CWRU 数据处理脚本
 - 遍历 data/CWRU_Bearing_Data 下的 16 个 .mat 文件
 - 提取 DE 信号
 - 小波去噪 + 信号级 Z-score 标准化
 - 滑窗分片 + 特征提取（13维）
 - 频率归一化（按源域标准或固定转速）
 - 特征级标准化（保存 mean/std）
 - 输出: stepX_CWRU_features.npy, stepX_CWRU_labels.npy, feature_scaler_CWRU.npz
"""

import os
import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from scipy.io import loadmat
from tqdm import tqdm

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

# -------------------- 小波熵 --------------------
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

# -------------------- 文件标签映射 --------------------
file_label_map = {
    # 正常
    "97.mat": 0, "98.mat": 0, "99.mat": 0, "100.mat": 0,
    # 内圈
    "105.mat": 1, "106.mat": 1, "107.mat": 1, "108.mat": 1,
    # 外圈
    "169.mat": 2, "170.mat": 2, "171.mat": 2, "172.mat": 2,
    # 滚动体
    "118.mat": 3, "119.mat": 3, "120.mat": 3, "121.mat": 3
}

# -------------------- 主流程 --------------------
def main():
    DATA_ROOT = os.path.join('data', 'CWRU_Bearing_Data')
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"{DATA_ROOT} 不存在")
    files = [f for f in os.listdir(DATA_ROOT) if f.endswith('.mat')]

    fs, win, step = 12000, 2048, 1024
    freq_cols = [6,7,8,9]

    X_all, y_all = [], []

    for f in tqdm(files, desc="处理 CWRU 文件"):
        file_path = os.path.join(DATA_ROOT, f)
        label = file_label_map.get(f, None)
        if label is None:
            continue
        data_dict = loadmat(file_path)
        # 取 DE 信号
        key = next((k for k in data_dict if k.endswith('DE_time')), None)
        if key is None:
            continue
        signal = data_dict[key].flatten()
        # 小波去噪
        signal = wavelet_denoise(signal)
        # 信号级 Z-score 标准化
        signal = (signal - np.mean(signal)) / (np.std(signal)+1e-6)
        # 滑窗分片 + 特征提取
        for start in range(0, len(signal)-win+1, step):
            sl = signal[start:start+win]
            feats = extract_features(sl, fs)
            X_all.append(feats)
            y_all.append(label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # 特征级 Z-score
    mean, std = np.mean(X_all, axis=0), np.std(X_all, axis=0)
    std[std==0] = 1.0
    X_std = (X_all - mean) / std

    # 保存
    np.save('stepX_CWRU_features.npy', X_std)
    np.save('stepX_CWRU_labels.npy', y_all)
    np.savez('feature_scaler_CWRU.npz', mean=mean, std=std)
    print(f"✅ CWRU 数据处理完成，特征矩阵: {X_std.shape}, 标签: {y_all.shape}")

if __name__ == "__main__":
    main()
