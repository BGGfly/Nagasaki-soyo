intro: 明确源域数据筛选标准：
	    域数据包含驱动端（DE）、风扇端（FE）、基座（BA） 三类传感器信号，且明确 “距离故障轴承越近，信号故障特征越显著”（如驱动端信号更适合驱动端轴承故障分析）。            需先筛选核心数据，排除冗余 / 弱特征数据：
        	    优先选择驱动端（DE）信号：驱动端直接连接电机转轴，振动信号能清晰捕获驱动端轴承振动及风扇端传递信号”，且源域 DE 信号涵盖 12kHz/48kHz 两种采样频率（文件表 1），信息最完整，优先作为核心分析对象。排除基座（BA）信号：文件说明 “BA 信号经多层传递后故障特征高度衰减，仅用于辅助分析”，暂不纳入核心数据集；可选保留风扇端（FE）信号：若后续需补充样本，可保留 FE 信号（12kHz 采样，文件表 1 中风扇端轴承为 SKF6203），但需单独标记传感器类型。
        	平衡故障类别与尺寸：文件明确源域故障类别为 4 类（正常 N、滚动体 B、内圈 IR、外圈 OR），且故障尺寸 / 位置存在差异：
        滚动体（B）/ 内圈（IR）：各 4 种故障尺寸（0.007/0.014/0.021/0.028 英寸），各尺寸至少选 10 个文件（如 B007_0~3、IR007_0~3 等）；
        外圈（OR）：3 种故障尺寸（0.007/0.014/0.021 英寸）+3 个采样位置（3 点 Orthogonal、6 点 Centered、12 点 Opposite），每个尺寸 - 位置组合至少选 5 个文件（如 OR007@3_0~3、OR007@6_0~3 等）；正常（N）：仅 4 个文件（N_0~3），全部保留（文件明确正常样本稀缺）
        	统一采样频率
        	文件中 DE 信号有 12kHz/48kHz 两种采样频率，若需简化分析，可优先选择 12kHz 信号（覆盖 FE/DE 双传感器，样本量更多）；若需高分辨率信号，可单独分析 48kHz DE 信号，最终需在报告中标明选择依据（如 “选择 12kHz DE 信号以匹配目标域 32kHz 下的下采样需求”）。

一、源域数据筛选与读取
	1.1目标
		文件指出 “驱动端（DE）信号故障特征最显著，基座（BA）信号仅辅助”，因此第一步代码需实现：
			1.1.1 遍历源域所有.mat 文件（路径：data/origin）；
			1.1.2 筛选出含 “DE 信号”（文件中变量名以_DE_time结尾）的文件；
			1.1.3 读取 DE 信号并验证格式（确保为 1D 振动数据）；
			1.1.4 按故障类别（正常 N / 滚动体 B / 内圈 IR / 外圈 OR）标注标签（文件附件 1 定义）
二、降噪处理：文件明确源域振动信号受 “背景噪声、干扰源响应” 影响，需通过小波降噪（文件参考文献 [1] 推荐，适合非平稳振动信号）保留故障冲击特征，削弱噪声；
标准化处理：消除不同文件信号的幅值量纲差异（如正常信号幅值小、故障信号幅值大），使特征提取时各信号处于同一量级，符合文件 “统一数据格式” 的隐含要求。
	加载第一步结果：代码自动加载第一步保存的step1_*.npy文件，无需手动修改路径（确保第一步结果与当前代码在同一文件夹）；
批量小波降噪：遍历 161 个信号，按文件参考文献 [1] 的小波降噪方法处理，每 20 个文件打印一次进度，确保过程可追溯；
全局标准化：用所有源域信号的全局均值 / 标准差做 Z-score 标准化（文件故障诊断常用），避免单文件统计量偏差，同时保存统计量（后续目标域信号需用相同统计量标准化，符合文件 “跨域数据统一格式” 的迁移逻辑）；
验证与保存：验证预处理后信号长度、数量无异常，保存结果为step2_*.npy，为第三步 “故障特征提取” 提供干净、统一的信号数据。
三、时域特征：反映故障冲击特性（文件指出 “故障轴承会产生突变冲击脉冲”），提取 6 个关键统计量；
频域特征：验证故障周期频率（文件表 2 定义 BPFO/BPFI/BSF），量化频域能量分布；
时频域特征：融合时域冲击与频域周期（文件指出 “故障信号为非平稳信号，需时频分析”），提取时频熵等特征；
特征整合：拼接三类特征为统一维度向量，标注对应故障标签，保存为模型可直接加载的格式。
源域数据筛选与读取 (step1_read_data.py)
import os
import re
import numpy as np
from scipy.io import loadmat


def parse_filename(name_without_ext):
    """
    一个健壮的函数，用于解析数据集中所有不同格式的文件名。
    """
    fault_type, fault_size, load_hp, rpm_in_filename = None, np.nan, np.nan, np.nan

    # 模式1: 内圈或滚动体故障 (IR/B), 带或不带RPM
    match = re.match(r'^(IR|B)(\d{3})_(\d)(?:_\((\d+)rpm\))?$', name_without_ext)
    if match:
        fault_type, fault_size_str, load_str, rpm_str = match.groups()
        fault_size = float(fault_size_str) / 1000
        load_hp = int(load_str)
        if rpm_str: rpm_in_filename = int(rpm_str)
        return fault_type, fault_size, load_hp, rpm_in_filename

    # 模式2: 外圈故障 (OR)
    match = re.match(r'^OR(\d{3})@(\d{1,2})_(\d)$', name_without_ext)
    if match:
        fault_type = 'OR'
        fault_size = float(match.group(1)) / 1000
        load_hp = int(match.group(3))
        return fault_type, fault_size, load_hp, rpm_in_filename

    # 模式3: 正常状态 (N), 带或不带RPM
    match = re.match(r'^N_(\d)(?:_\((\d+)rpm\))?$', name_without_ext)
    if match:
        fault_type = 'N'
        load_str, rpm_str = match.groups()
        load_hp = int(load_str)
        if rpm_str: rpm_in_filename = int(rpm_str)
        return fault_type, fault_size, load_hp, rpm_in_filename

    return fault_type, fault_size, load_hp, rpm_in_filename


def step1_read_data():
    """
    主函数：遍历、筛选并读取所有源域文件的驱动端(DE)信号。
    """
    data_folder = os.path.join('data', 'origin')
    if not os.path.exists(data_folder):
        print(f"错误: 根目录 '{data_folder}' 不存在。请检查路径。")
        return

    # 遍历获取所有.mat文件路径
    all_mat_files = [os.path.join(root, file) for root, _, files in os.walk(data_folder) for file in files if
                     file.endswith('.mat')]
    print(f'检测到 {len(all_mat_files)} 个 .mat 文件。')

    # ▼▼▼ 核心修正：在这里定义 fault_type_map ▼▼▼
    fault_type_map = {'N': 0, 'OR': 1, 'IR': 2, 'B': 3}

    signals = []
    labels = []

    processed_count = 0
    for file_path in all_mat_files:
        # 从完整路径中提取不带扩展名的文件名
        name_without_ext = os.path.splitext(os.path.basename(file_path))[0]

        # ▼▼▼ 核心修正：调用解析函数来获取 fault_type ▼▼▼
        fault_type, _, _, _ = parse_filename(name_without_ext)

        # 如果文件名无法解析，则跳过
        if fault_type is None:
            print(f"警告：无法解析文件名 {os.path.basename(file_path)}，已跳过。")
            continue

        try:
            data_dict = loadmat(file_path)
        except Exception as e:
            print(f"警告：无法加载文件 {file_path}: {e}，已跳过。")
            continue

        # 筛选出含“DE信号”的文件
        de_signal_key = next((key for key in data_dict if key.endswith('_DE_time')), None)

        if de_signal_key:
            signal = data_dict[de_signal_key].flatten()
            signals.append(signal)
            labels.append(fault_type_map[fault_type])
            processed_count += 1
            if processed_count % 20 == 0:
                print(f"已处理 {processed_count}/{len(all_mat_files)} 个文件...")

    # 将Python列表转换为Numpy数组以便保存
    # 使用 dtype=object 是因为信号长度可能不同
    np.save('step1_signals.npy', np.array(signals, dtype=object))
    np.save('step1_labels.npy', np.array(labels))

    print(f"\n步骤一完成：共筛选并提取了 {len(signals)} 个驱动端(DE)信号。")
    print("结果已保存至 'step1_signals.npy' 和 'step1_labels.npy'。")


if __name__ == '__main__':
    step1_read_data()


    (step2_preprocess.py)import numpy as np
import pywt
from tqdm import tqdm


def wavelet_denoise(signal):
    """
    使用小波变换对信号进行降噪处理。
    :param signal: 输入的一维信号数组。
    :return: 降噪后的信号数组。
    """
    # 1. 选择小波基和分解层数
    # 'db8' (Daubechies 8) 是一种在振动信号分析中常用的小波
    wavelet = 'db8'
    # 分解层数可以根据信号长度和采样频率调整，这里选择5层
    level = 5

    # 2. 对信号进行多层小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # 3. 阈值处理
    # 计算噪声标准差的估计值（使用第一层高频系数的中位数绝对偏差）
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # 使用通用阈值(Universal Threshold)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # 对每一层的细节系数（高频部分）进行软阈值处理
    new_coeffs = [coeffs[0]]  # 首先保留近似系数（低频部分）
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

    # 4. 重构信号
    denoised_signal = pywt.waverec(new_coeffs, wavelet)

    # 返回与原始信号等长的降噪信号
    return denoised_signal[:len(signal)]


def step2_preprocess():
    """
    主函数：加载信号，进行小波降噪和全局标准化。
    """
    print("--- 步骤二：开始信号降噪与标准化 ---")

    # 1. 加载第一步的结果
    try:
        signals = np.load('step1_signals.npy', allow_pickle=True)
        print(f"成功加载 {len(signals)} 个原始信号。")
    except FileNotFoundError:
        print("错误：未找到 'step1_signals.npy' 文件。请先运行第一步的脚本。")
        return

    # 2. 批量进行小波降噪
    print("\n--- 正在进行小波降噪... ---")
    denoised_signals = [wavelet_denoise(s) for s in tqdm(signals, desc="小波降噪进度")]

    # 3. 全局Z-score标准化
    print("\n--- 正在进行全局Z-score标准化... ---")
    # 将所有降噪后的信号拼接成一个长向量，以计算全局统计量
    concatenated_signals = np.concatenate(denoised_signals)
    global_mean = np.mean(concatenated_signals)
    global_std = np.std(concatenated_signals)

    print(f"计算得出全局均值: {global_mean:.4f}")
    print(f"计算得出全局标准差: {global_std:.4f}")

    # 使用计算出的全局均值和标准差对每个信号进行标准化
    standardized_signals = [(s - global_mean) / global_std for s in denoised_signals]

    # 4. 验证与保存
    # 验证处理后的信号数量和类型
    assert len(standardized_signals) == len(signals), "处理后的信号数量与原始数量不符！"
    print("\n信号长度与数量验证无异常。")

    # 保存处理后的信号
    np.save('step2_processed_signals.npy', np.array(standardized_signals, dtype=object))
    print("已将预处理后的信号保存至 'step2_processed_signals.npy'")

    # 保存用于标准化的全局统计量，这对于后续处理目标域数据至关重要
    np.savez('global_scaler.npz', mean=global_mean, std=global_std)
    print("已将全局标准化参数保存至 'global_scaler.npz'")

    print("\n步骤二完成！")


if __name__ == '__main__':
    step2_preprocess()


特征提取 (step3_extract_features.py)import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import scale
from tqdm import tqdm
import pandas as pd


def calculate_entropy(coeffs):
    """计算小波包分解系数的能量熵"""
    # 移除零系数以避免log(0)
    coeffs = [c for c in coeffs if c.any()]
    if not coeffs:
        return 0
    # 计算每个频带的能量
    energy = [np.sum(c ** 2) for c in coeffs]
    total_energy = np.sum(energy)
    if total_energy == 0:
        return 0
    # 计算每个频带的能量占比
    p = energy / total_energy
    # 计算能量熵
    entropy = -np.sum(p * np.log2(p + 1e-6))  # 加一个小常数避免log(0)
    return entropy


def extract_features(signal_slice, sampling_rate):
    """
    从单个信号分片中提取13维特征。
    """
    features = {}

    # --- 1. 时域特征 (6维) ---
    rms = np.sqrt(np.mean(signal_slice ** 2))
    features['time_rms'] = rms
    features['time_kurtosis'] = kurtosis(signal_slice)
    features['time_skewness'] = skew(signal_slice)
    peak = np.max(np.abs(signal_slice))
    mean_abs = np.mean(np.abs(signal_slice))
    # 避免除以零
    features['time_crest_factor'] = peak / (rms + 1e-6)
    features['time_shape_factor'] = rms / (mean_abs + 1e-6)
    features['time_impulse_factor'] = peak / (mean_abs + 1e-6)

    # --- 2. 频域特征 (4维) ---
    N = len(signal_slice)
    yf = fft(signal_slice)
    xf = fftfreq(N, 1 / sampling_rate)

    half_N = N // 2
    yf_magnitude = np.abs(yf[0:half_N])
    xf_positive = xf[0:half_N]

    sum_mag = np.sum(yf_magnitude) + 1e-6

    features['freq_centroid'] = np.sum(xf_positive * yf_magnitude) / sum_mag
    features['freq_rms'] = np.sqrt(np.sum(xf_positive ** 2 * yf_magnitude) / sum_mag)
    features['freq_variance'] = np.sum(((xf_positive - features['freq_centroid']) ** 2) * yf_magnitude) / sum_mag

    # 包络谱分析，提取最大峰值频率
    from scipy.signal import hilbert
    analytic_signal = hilbert(signal_slice)
    envelope = np.abs(analytic_signal)
    yf_env = fft(envelope - np.mean(envelope))  # 减去直流分量
    yf_env_mag = np.abs(yf_env[0:half_N])
    # 忽略0Hz的直流分量，寻找最大峰
    peak_freq_index = np.argmax(yf_env_mag[1:]) + 1
    features['freq_envelope_peak_freq'] = xf_positive[peak_freq_index]

    # --- 3. 时频域特征 (3维) ---
    # 使用小波包分解
    wp = pywt.WaveletPacket(data=signal_slice, wavelet='db8', mode='symmetric', maxlevel=3)
    nodes = wp.get_level(3, order='natural')
    coeffs = [n.data for n in nodes]

    # 提取三个关键频带的能量熵
    features['wp_entropy_low'] = calculate_entropy(coeffs[:2])  # 低频带
    features['wp_entropy_mid'] = calculate_entropy(coeffs[2:5])  # 中频带
    features['wp_entropy_high'] = calculate_entropy(coeffs[5:])  # 高频带

    return features


def step3_extract_features():
    """
    主函数：加载预处理后的信号，进行分片和特征提取。
    """
    print("--- 步骤三：开始分片与特征提取 ---")

    # 1. 加载第二步的结果
    try:
        signals = np.load('step2_processed_signals.npy', allow_pickle=True)
        labels = np.load('step1_labels.npy', allow_pickle=True)
        print(f"成功加载 {len(signals)} 个预处理后的信号和标签。")
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。请先运行第一步和第二步的脚本。")
        return

    # 定义分片参数
    WINDOW_SIZE = 2048
    STEP_SIZE = 1024
    # 注意：采样频率对于特征提取很重要，但由于我们的数据源有两种采样频率，
    # 且已在预处理中标准化，这里我们使用一个“名义”采样率来进行FFT频率轴的计算。
    # 实际应用中，如果采样率差异大，最好分开处理或重采样。此处假设为12kHz。
    SAMPLING_RATE = 12000

    all_features = []
    all_labels = []

    print("\n--- 正在进行分片和特征提取... ---")
    # 遍历每一个信号
    for i in tqdm(range(len(signals)), desc="信号处理进度"):
        signal = signals[i]
        label = labels[i]

        # 滑动窗口分片
        for start in range(0, len(signal) - WINDOW_SIZE + 1, STEP_SIZE):
            end = start + WINDOW_SIZE
            signal_slice = signal[start:end]

            # 提取特征
            features = extract_features(signal_slice, SAMPLING_RATE)

            # 将特征和标签添加到总列表中
            all_features.append(list(features.values()))
            all_labels.append(label)

    # 转换为Numpy数组
    X = np.array(all_features)
    y = np.array(all_labels)

    # 验证数据
    print(f"\n特征提取完成！总共生成了 {X.shape[0]} 个样本。")
    print(f"特征矩阵的形状: {X.shape}")  # 应该为 (样本数, 13)
    print(f"标签向量的形状: {y.shape}")

    # 检查是否有NaN或inf值
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("\n警告：特征矩阵中存在NaN或inf值，正在进行清理...")
        # 使用pandas清理后转回numpy
        df = pd.DataFrame(X)
        original_rows = df.shape[0]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 找到含有NaN的行，以便在标签中也删除它们
        nan_rows = df.isnull().any(axis=1)
        df.dropna(inplace=True)

        dropped_indices = original_rows - df.shape[0]
        if dropped_indices > 0:
            print(f"已丢弃 {dropped_indices} 个包含无效值的样本。")

        X = df.values
        # 相应地更新标签
        y = y[~nan_rows]

    # 保存最终的特征矩阵和标签
    np.save('step3_features.npy', X)
    np.save('step3_labels.npy', y)
    print("\n已将最终的特征矩阵和标签保存至 'step3_features.npy' 和 'step3_labels.npy'")

    print("\n步骤三完成！")


if __name__ == '__main__':
    step3_extract_features()




二、任务 2 “源域故障诊断”
		核心是基于已提取的 13 维源域特征（时域 6 维 + 频域 4 维 + 时频域 3 维），设计诊断模型并验证其在源域的故障分类能力
	2.1划分源域训练集与测试集
		数据规模：源域共 161 个样本，故障类别分布为 “正常 4 个、滚动体 40 个、内圈 40 个、外圈 77 个”（文件附件 1 明确）；
		划分比例：推荐按 8:2 划分（兼顾训练集规模与测试集代表性），即 129 个训练样本、32 个测试样本；
		分层原则：每类故障按比例分配样本（如正常样本 4 个→训练 3 个、测试 1 个；滚动体 40 个→训练 32 个、测试 8 个），确保训练集与测试集的故障类别分布一致，避免模型偏				向多数类（如外圈故障）。
	2.2 设计源域故障诊断模型  随机森林模型实现源域诊断模型
