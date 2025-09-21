import os
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

# 定义数据文件所在的根目录
#data_folder = r'C:\Users\12394\Desktop\数据集\数据集\源域数据集'  # 确认这个路径！

data_folder = os.path.join('data', 'origin')

# --- 添加以下两行代码进行调试 ---
print(f"尝试搜索的根目录: {data_folder}")
if not os.path.exists(data_folder):
    print(f"错误: 根目录 '{data_folder}' 不存在。请检查路径。")
    exit()  # 如果目录不存在，直接退出


# --- 调试代码结束 ---


# --- 定义文件名解析函数 ---
def parse_filename(name_without_ext):
    fault_type = None
    fault_size = np.nan
    load_hp = np.nan
    rpm_in_filename = np.nan  # 从文件名中解析出的RPM (如果存在)

    # 模式1: 内圈或滚动体故障 (IR/B)，可能带或不带RPM信息
    # IR/B尺寸_载荷_(RPM) -> 例如 B028_0_(1797rpm)
    match_ir_b_load_rpm = re.match(r'^(IR|B)(\d{3})_(\d)_\((\d+)rpm\)$', name_without_ext)
    if match_ir_b_load_rpm:
        fault_type = match_ir_b_load_rpm.group(1)
        fault_size = float(match_ir_b_load_rpm.group(2)) / 1000
        load_hp = int(match_ir_b_load_rpm.group(3))
        rpm_in_filename = int(match_ir_b_load_rpm.group(4))
        return fault_type, fault_size, load_hp, rpm_in_filename

    # IR/B尺寸_载荷 -> 例如 B007_0
    match_ir_b_load = re.match(r'^(IR|B)(\d{3})_(\d)$', name_without_ext)
    if match_ir_b_load:
        fault_type = match_ir_b_load.group(1)
        fault_size = float(match_ir_b_load.group(2)) / 1000
        load_hp = int(match_ir_b_load.group(3))
        return fault_type, fault_size, load_hp, rpm_in_filename

    # 模式2: 外圈故障 (OR)，尺寸@位置_载荷
    # OR尺寸@位置_载荷 -> 例如 OR007@6_0
    match_or = re.match(r'^OR(\d{3})@(\d{1,2})_(\d)$', name_without_ext)
    if match_or:
        fault_type = 'OR'
        fault_size = float(match_or.group(1)) / 1000
        # outer_race_pos = int(match_or.group(2)) # 外圈故障位置，目前暂时不存入DataFrame，但可以提取
        load_hp = int(match_or.group(3))
        return fault_type, fault_size, load_hp, rpm_in_filename

    # 模式3: 正常状态 (N)，可能带或不带RPM信息
    # N_载荷_(RPM) -> 例如 N_1_(1772rpm)
    match_n_load_rpm = re.match(r'^N_(\d)_\((\d+)rpm\)$', name_without_ext)
    if match_n_load_rpm:
        fault_type = 'N'
        load_hp = int(match_n_load_rpm.group(1))
        rpm_in_filename = int(match_n_load_rpm.group(2))
        return fault_type, fault_size, load_hp, rpm_in_filename

    # N_载荷 -> 例如 N_0
    match_n_load = re.match(r'^N_(\d)$', name_without_ext)
    if match_n_load:
        fault_type = 'N'
        load_hp = int(match_n_load.group(1))
        return fault_type, fault_size, load_hp, rpm_in_filename

    # 如果所有模式都不匹配
    return fault_type, fault_size, load_hp, rpm_in_filename


# --- 文件遍历和信息提取 ---
all_mat_files = []
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.mat'):
            all_mat_files.append(os.path.join(root, file))

print(f'检测到 {len(all_mat_files)} 个 .mat 文件。\n')

# 初始化一个列表来存储所有文件的信息字典
all_data_info_list = []

# 定义故障类型映射
fault_type_map = {'N': 0, 'OR': 1, 'IR': 2, 'B': 3}

# 遍历文件列表，提取信息
for i, file_path in enumerate(all_mat_files):
    file_name_with_ext = os.path.basename(file_path)
    name_without_ext = os.path.splitext(file_name_with_ext)[0]

    fault_type, fault_size, load_hp, rpm_in_filename_val = parse_filename(name_without_ext)

    if fault_type is not None:  # 如果文件名解析成功
        try:
            data_dict = loadmat(file_path)
        except Exception as e:
            print(f'警告: 无法加载文件 {file_path}: {e}')
            continue

        rpm_val_from_mat = np.nan
        sampling_freq = np.nan
        bearing_location = 'Unknown'

        for key in data_dict.keys():
            if 'RPM' in key and isinstance(data_dict[key], np.ndarray) and data_dict[key].size == 1:
                rpm_val_from_mat = data_dict[key].item()
                break

        rpm_final_val = rpm_val_from_mat
        if np.isnan(rpm_final_val) and not np.isnan(rpm_in_filename_val):
            rpm_final_val = rpm_in_filename_val

        # =================================================================
        # ▼▼▼ 核心修正部分 ▼▼▼
        # =================================================================
        normalized_path = os.path.normpath(file_path)
        path_parts = normalized_path.split(os.sep)

        for part in path_parts:
            part_lower = part.lower()
            # 只要文件夹名称中包含 'khz'，我们就处理它
            if 'khz' in part_lower:
                # 1. 提取采样频率
                match_freq = re.search(r'(\d+)khz', part_lower)
                if match_freq:
                    sampling_freq = int(match_freq.group(1)) * 1000

                # 2. 如果存在，再提取轴承位置
                if 'de_data' in part_lower:
                    bearing_location = 'DE'
                elif 'fe_data' in part_lower:
                    bearing_location = 'FE'

                # 找到并处理完包含khz的文件夹后，就可以停止对路径的搜索了
                break
        # =================================================================
        # ▲▲▲ 修正结束 ▲▲▲
        # =================================================================

        all_data_info_list.append({
            'FilePath': file_path,
            'FileName': file_name_with_ext,
            'FaultType_Str': fault_type,
            'FaultSize_inch': fault_size,
            'Load_HP': load_hp,
            'RPM': rpm_final_val,
            'SamplingFreq_Hz': sampling_freq,
            'BearingLocation': bearing_location,
            'FaultType_Label': fault_type_map[fault_type]
        })
    else:
        print(f'警告: 文件名格式不匹配，无法解析: {file_name_with_ext}')

# 将列表转换为 Pandas DataFrame 便于查看和操作
source_data_df = pd.DataFrame(all_data_info_list)

# 显示前几行数据表
print('\n源域数据信息预览:\n')
print(source_data_df.head())

# 统计各类故障数量
print('\n各类故障样本数量统计:\n')
print(source_data_df['FaultType_Str'].value_counts())

# 统计不同采样频率的数量
print('\n不同采样频率样本数量统计:\n')
print(source_data_df['SamplingFreq_Hz'].value_counts())

# 统计不同轴承位置的样本数量
print('\n不同轴承位置样本数量统计:\n')
print(source_data_df['BearingLocation'].value_counts())

# =========================================================================
# T2.1: 目标域数据初步了解 (无需加载，只需根据描述进行整理)
# =========================================================================

print('\n' + '-' * 40)
print('目标域数据集信息回顾:')
print('包含列车滚动轴承外圈（OR）、内圈（IR）、滚动体（B）故障和正常状态（N）下的振动信号数据。')
print('采集时间为8秒，采样频率为32kHz。')
print('列车速度约90km/h（轴承转速约600 rpm）。')
print('数据文件以英文字母A~P编号命名，各数据所属工作状态未知。')
print('-' * 40 + '\n')

# (可选) 将DataFrame保存到CSV文件，以便后续快速加载，无需重复解析
source_data_df.to_csv('source_data_metadata.csv', index=False)
print('源域数据元信息已保存到 source_data_metadata.csv')