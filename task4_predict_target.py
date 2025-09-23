import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from scipy.io import loadmat
import joblib
from tqdm import tqdm
from collections import Counter

# -------------------- 参数 --------------------
DATA_ROOT = os.path.join('data', 'target')
MODEL_FILE = 'task3_source_model.pkl'
SCALER_FILE = 'feature_scaler.npz'
OUTPUT_DIR = os.path.join('figures', 'task4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    'time_rms','time_kurtosis','time_skewness','time_crest_factor',
    'time_shape_factor','time_impulse_factor','freq_centroid','freq_rms',
    'freq_variance','freq_envelope_peak_freq','wp_entropy_low','wp_entropy_mid','wp_entropy_high'
]

LABEL_MAP = {0:'N',1:'IR',2:'OR',3:'B'}
fs, win, step = 12000, 2048, 1024
FREQ_COLS = [6,7,8,9]

# -------------------- 小波去噪 --------------------
def wavelet_denoise(signal):
    wavelet, level = 'db8',5
    coeffs = pywt.wavedec(signal,wavelet,level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2*np.log(len(signal)))
    new_coeffs = [coeffs[0]] + [pywt.threshold(c,threshold,mode='soft') for c in coeffs[1:]]
    return pywt.waverec(new_coeffs,wavelet)[:len(signal)]

# -------------------- 小波熵 --------------------
def calculate_entropy(coeffs):
    coeffs = [c for c in coeffs if c.any()]
    if not coeffs: return 0
    energy = [np.sum(c**2) for c in coeffs]
    total = np.sum(energy)
    if total==0: return 0
    p = energy/total
    return -np.sum(p*np.log2(p+1e-6))

# -------------------- 特征提取 --------------------
def extract_features(x, fs):
    feats = {}
    rms = np.sqrt(np.mean(x**2))
    peak, mean_abs = np.max(np.abs(x)), np.mean(np.abs(x))
    feats['time_rms']=rms
    feats['time_kurtosis']=kurtosis(x)
    feats['time_skewness']=skew(x)
    feats['time_crest_factor']=peak/(rms+1e-6)
    feats['time_shape_factor']=rms/(mean_abs+1e-6)
    feats['time_impulse_factor']=peak/(mean_abs+1e-6)
    N=len(x)
    yf, xf=fft(x), fftfreq(N,1/fs)
    half=N//2
    mag, freqs = np.abs(yf[:half]), xf[:half]
    sum_mag=np.sum(mag)+1e-6
    feats['freq_centroid']=np.sum(freqs*mag)/sum_mag
    feats['freq_rms']=np.sqrt(np.sum(freqs**2*mag)/sum_mag)
    feats['freq_variance']=np.sum(((freqs-feats['freq_centroid'])**2)*mag)/sum_mag
    env=np.abs(hilbert(x))
    env_fft=np.abs(fft(env-np.mean(env)))[:half]
    peak_idx=np.argmax(env_fft[1:])+1
    feats['freq_envelope_peak_freq']=freqs[peak_idx]
    wp=pywt.WaveletPacket(data=x,wavelet='db8',maxlevel=3)
    nodes=wp.get_level(3,order='natural')
    coeffs=[n.data for n in nodes]
    feats['wp_entropy_low']=calculate_entropy(coeffs[:2])
    feats['wp_entropy_mid']=calculate_entropy(coeffs[2:5])
    feats['wp_entropy_high']=calculate_entropy(coeffs[5:])
    return list(feats.values())

# -------------------- 主流程 --------------------
def main():
    clf = joblib.load(MODEL_FILE)
    scaler = np.load(SCALER_FILE)
    mean_src, std_src = scaler['mean'], scaler['std']
    std_src[std_src==0]=1.0

    all_files = sorted([f for f in os.listdir(DATA_ROOT) if f.endswith('.mat')])
    predictions = {}
    predictions_proba = {}

    for f in tqdm(all_files, desc="预测目标域样本"):
        data_dict = loadmat(os.path.join(DATA_ROOT,f))
        key = next(k for k in data_dict if k[0].isalpha())
        signal = data_dict[key].flatten()
        signal = wavelet_denoise(signal)
        signal = (signal - np.mean(signal))/(np.std(signal)+1e-6)

        # 滑窗特征提取
        X_sample=[]
        for start in range(0,len(signal)-win+1,step):
            sl = signal[start:start+win]
            feats = extract_features(sl,fs)
            X_sample.append(feats)
        X_sample = np.array(X_sample)
        X_sample_std = (X_sample - mean_src)/std_src

        # 预测概率
        proba = clf.predict_proba(X_sample_std)
        avg_proba = np.mean(proba,axis=0)  # 取平均
        predictions_proba[f] = avg_proba
        pred_label = np.argmax(avg_proba)
        predictions[f] = LABEL_MAP[pred_label]

    # 保存预测结果
    np.save(os.path.join(OUTPUT_DIR,'target_pred_labels.npy'), predictions)
    np.save(os.path.join(OUTPUT_DIR,'target_pred_proba.npy'), predictions_proba)

    print("✅ 预测完成，结果如下:")
    for k,v in predictions.items():
        print(f"  {k}: {v}")

    # 绘制柱状图
    plt.figure(figsize=(12,6))
    files = list(predictions_proba.keys())
    probs = np.array([predictions_proba[f] for f in files])
    bar_width = 0.2
    x = np.arange(len(files))
    for i, label in LABEL_MAP.items():
        plt.bar(x + i*bar_width, probs[:,i], width=bar_width, label=label)
    plt.xticks(x + 1.5*bar_width, files, rotation=45)
    plt.ylabel('Prediction Probability')
    plt.title('Target Samples Prediction Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'target_prediction_proba.png'))
    plt.close()
    print(f"✅ 柱状图已保存到 {OUTPUT_DIR}/target_prediction_proba.png")

if __name__=="__main__":
    main()
