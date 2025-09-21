import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
from tqdm import tqdm  # 引入tqdm来显示进度条

# --- 1. 定义分片和特征提取参数 ---

# 分片参数
WINDOW_SIZE = 2048  # 窗口大小
STEP_SIZE = 1024  # 步长 (50%重叠)


# 特征提取函数 (与上一版基本相同)
def extract_signal_features(signal, fs):
    features = {}
    signal = np.asarray(signal).flatten()

    # 时域特征
    features['time_mean'] = np.mean(signal)
    features['time_std'] = np.std(signal)
    features['time_rms'] = np.sqrt(np.mean(signal ** 2))
    features['time_peak'] = np.max(np.abs(signal))
    features['time_p2p'] = np.max(signal) - np.min(signal)
    features['time_kurtosis'] = kurtosis(signal)
    features['time_skewness'] = skew(signal)

    mean_abs_value = np.mean(np.abs(signal))
    # 防止分母为零
    if features['time_rms'] != 0:
        features['time_crest_factor'] = features['time_peak'] / features['time_rms']
    else:
        features['time_crest_factor'] = np.nan

    if mean_abs_value != 0:
        features['time_shape_factor'] = features['time_rms'] / mean_abs_value
        features['time_impulse_factor'] = features['time_peak'] / mean_abs_value
    else:
        features['time_shape_factor'] = np.nan
        features['time_impulse_factor'] = np.nan

    # 频域特征
    N = len(signal)
    if N == 0: return {**features, 'freq_centroid': np.nan, 'freq_rms': np.nan, 'freq_variance': np.nan}

    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)

    half_N = N // 2
    yf_magnitude = np.abs(yf[0:half_N])
    xf_positive = xf[0:half_N]

    if np.sum(yf_magnitude) == 0:
        return {**features, 'freq_centroid': np.nan, 'freq_rms': np.nan, 'freq_variance': np.nan}

    features['freq_centroid'] = np.sum(xf_positive * yf_magnitude) / np.sum(yf_magnitude)
    features['freq_rms'] = np.sqrt(np.sum(xf_positive ** 2 * yf_magnitude) / np.sum(yf_magnitude))
    features['freq_variance'] = np.sum(((xf_positive - features['freq_centroid']) ** 2) * yf_magnitude) / np.sum(
        yf_magnitude)

    return features


# --- 2. 主程序流程 ---

# 加载之前整理好的元数据
try:
    metadata_df = pd.read_csv('source_data_metadata.csv')
    print("成功加载元数据文件 'source_data_metadata.csv'")
except FileNotFoundError:
    print("错误: 未找到 'source_data_metadata.csv'。请先运行上一步的脚本。")
    exit()

all_features_list = []

# 使用tqdm来包装迭代器，以显示进度条
print("开始分片并提取特征...")
for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
    file_path = row['FilePath']
    sampling_freq = row['SamplingFreq_Hz']

    # 加载.mat文件
    try:
        data_dict = loadmat(file_path)
    except Exception as e:
        print(f'\n警告: 无法加载文件 {file_path}: {e}')
        continue

    # 优先选择驱动端(DE)信号，其次是风扇端(FE)
    signal_data = None
    for key in data_dict.keys():
        if 'DE_time' in key:
            signal_data = data_dict[key].flatten()
            break
        elif 'FE_time' in key:
            signal_data = data_dict[key].flatten()
            break

    # 确认信号和采样频率有效
    if signal_data is None or np.isnan(sampling_freq):
        print(f"\n警告: 文件 {row['FileName']} 缺少振动信号或采样频率。")
        continue

    # --- 核心：滑动窗口分片循环 ---
    num_segments = 0
    for start_idx in range(0, len(signal_data) - WINDOW_SIZE + 1, STEP_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        segment = signal_data[start_idx:end_idx]

        if len(segment) == WINDOW_SIZE:
            # 对当前分片提取特征
            extracted_features = extract_signal_features(segment, sampling_freq)

            # 继承原始文件的元数据
            segment_metadata = row.to_dict()

            # 将提取的特征合并到字典中
            segment_metadata.update(extracted_features)

            # 添加到总列表中
            all_features_list.append(segment_metadata)
            num_segments += 1

# --- 3. 创建并保存最终的特征DataFrame ---
final_features_df = pd.DataFrame(all_features_list)

print(f"\n处理完成！")
print(f"从 {metadata_df.shape[0]} 个文件中，总共生成了 {final_features_df.shape[0]} 个样本。")

# (可选) 检查是否有NaN值
if final_features_df.isnull().sum().sum() > 0:
    print("\n警告: 特征数据中存在NaN值，请检查。")
    print(final_features_df.isnull().sum())

# 保存包含所有分片特征的CSV文件
output_csv_path = 'source_data_sliced_features.csv'
final_features_df.to_csv(output_csv_path, index=False)
print(f'\n所有源域数据的分片特征已保存到 {output_csv_path}')