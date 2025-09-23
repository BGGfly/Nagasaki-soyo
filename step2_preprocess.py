"""
Step2: 源域信号预处理
 - 小波去噪
 - 全局 Z-score 标准化
 - 保存 step2_processed_signals.npy 和 global_scaler.npz
"""

import numpy as np
import pywt
from tqdm import tqdm


def wavelet_denoise(signal):
    wavelet, level = 'db8', 5
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    return pywt.waverec(new_coeffs, wavelet)[:len(signal)]


def step2_preprocess():
    try:
        signals = np.load('step1_signals.npy', allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError("缺少 step1_signals.npy，请先运行 step1")

    print(f"加载 {len(signals)} 条信号")

    denoised = [wavelet_denoise(s) for s in tqdm(signals, desc="小波去噪")]
    concat = np.concatenate(denoised)
    mean, std = np.mean(concat), np.std(concat)

    standardized = [(s - mean) / std for s in denoised]
    np.save('step2_processed_signals.npy', np.array(standardized, dtype=object))
    np.savez('global_scaler.npz', mean=mean, std=std)

    print(f"✅ Step2 完成，已保存 step2_processed_signals.npy 和 global_scaler.npz")


if __name__ == "__main__":
    step2_preprocess()
