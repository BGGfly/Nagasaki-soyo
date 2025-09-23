"""
Step3: 源域特征提取
 - 滑窗分片 (2048, 步长1024)
 - 提取 13 维特征 (时域/频域/时频域)
 - 保存 step3_features.npy 和 step3_labels.npy
"""

import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from tqdm import tqdm
import pandas as pd


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
    return feats


def step3_extract_features():
    try:
        signals = np.load('step2_processed_signals.npy', allow_pickle=True)
        labels = np.load('step1_labels.npy', allow_pickle=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"缺少文件: {e.filename}")

    fs, win, step = 12000, 2048, 1024
    X, y = [], []

    for i in tqdm(range(len(signals)), desc="提取特征"):
        sig, label = signals[i], labels[i]
        for start in range(0, len(sig) - win + 1, step):
            sl = sig[start:start+win]
            feats = extract_features(sl, fs)
            X.append(list(feats.values()))
            y.append(label)

    X, y = np.array(X), np.array(y)

    # 清理 NaN / inf
    df = pd.DataFrame(X)
    nan_rows = df.isnull().any(axis=1)
    if nan_rows.any():
        print(f"⚠️ 移除 {nan_rows.sum()} 行 NaN/inf 特征")
        df = df.dropna()
        X, y = df.values, y[~nan_rows]

    np.save('step3_features.npy', X)
    np.save('step3_labels.npy', y)
    print(f"✅ Step3 完成，特征矩阵: {X.shape}, 标签: {y.shape}")


if __name__ == "__main__":
    step3_extract_features()
