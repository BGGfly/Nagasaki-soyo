"""
Step1: 源域数据读取与筛选
 - 遍历 data/origin 下的 .mat 文件
 - 解析文件名提取故障类型
 - 只提取驱动端(DE)信号
 - 保存 step1_signals.npy, step1_labels.npy
"""

import os
import re
import numpy as np
from scipy.io import loadmat


def parse_filename(name_without_ext):
    """解析文件名，返回 (fault_type, fault_size, load_hp, rpm)"""
    fault_type, fault_size, load_hp, rpm_in_filename = None, np.nan, np.nan, np.nan

    # IR / B (含可选 RPM)
    match = re.match(r'^(IR|B)(\d{3})_(\d)(?:_\((\d+)rpm\))?$', name_without_ext)
    if match:
        fault_type, size_str, load_str, rpm_str = match.groups()
        fault_size = float(size_str) / 1000
        load_hp = int(load_str)
        if rpm_str:
            rpm_in_filename = int(rpm_str)
        return fault_type, fault_size, load_hp, rpm_in_filename

    # OR
    match = re.match(r'^OR(\d{3})@(\d{1,2})_(\d)$', name_without_ext)
    if match:
        fault_type = 'OR'
        fault_size = float(match.group(1)) / 1000
        load_hp = int(match.group(3))
        return fault_type, fault_size, load_hp, rpm_in_filename

    # Normal
    match = re.match(r'^N_(\d)(?:_\((\d+)rpm\))?$', name_without_ext)
    if match:
        fault_type = 'N'
        load_str, rpm_str = match.groups()
        load_hp = int(load_str)
        if rpm_str:
            rpm_in_filename = int(rpm_str)
        return fault_type, fault_size, load_hp, rpm_in_filename

    return fault_type, fault_size, load_hp, rpm_in_filename


def step1_read_data():
    data_folder = os.path.join('data', 'origin')
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"未找到源域数据目录: {data_folder}")

    all_mat_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(data_folder)
        for f in files if f.endswith('.mat')
    ]
    print(f"检测到 {len(all_mat_files)} 个 .mat 文件")

    fault_type_map = {'N': 0, 'OR': 1, 'IR': 2, 'B': 3}

    signals, labels = [], []

    for file_path in all_mat_files:
        name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
        fault_type, _, _, _ = parse_filename(name_without_ext)

        if fault_type is None:
            print(f"⚠️ 跳过无法解析的文件: {file_path}")
            continue

        try:
            data_dict = loadmat(file_path)
        except Exception as e:
            print(f"⚠️ 加载失败 {file_path}: {e}")
            continue

        # 提取 DE 信号
        de_key = next((k for k in data_dict if k.endswith('_DE_time')), None)
        if de_key:
            signals.append(data_dict[de_key].flatten())
            labels.append(fault_type_map[fault_type])

    np.save('step1_signals.npy', np.array(signals, dtype=object))
    np.save('step1_labels.npy', np.array(labels))
    print(f"✅ Step1 完成，共提取 {len(signals)} 条 DE 信号")


if __name__ == "__main__":
    step1_read_data()
