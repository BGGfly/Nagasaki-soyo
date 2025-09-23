import os
import re
import urllib.request
import zipfile
from scipy.io import loadmat
import numpy as np

# -------------------- 参数 --------------------
DATA_DIR = os.path.join('data', 'extra')
CWRU_URLS = [
    # 这里列出官方 CWRU 数据下载链接（可根据需要增删）
    'https://github.com/CWRU-Fault-Dataset/drive_end/12kHz/12k_drive_end.zip',
    # 其他链接可以继续添加
]

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# -------------------- 下载并解压 --------------------
for url in CWRU_URLS:
    filename = url.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"⬇️ 下载 {filename}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"✅ 已存在 {filename}")

    # 解压
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print(f"✅ 解压 {filename}")


# -------------------- 解析标签 --------------------
def parse_cwru_filename(name):
    """解析文件名，返回 (fault_type, fault_size, load_hp, rpm)"""
    fault_type, fault_size, load_hp, rpm = None, np.nan, np.nan, np.nan
    # 内圈/外圈/滚动体/正常
    match = re.match(r'^(IR|OR|B|N)(\d{3})?_?(\d)?_?(\d+)?', name)
    if match:
        fault_type, size_str, load_str, rpm_str = match.groups()
        if size_str: fault_size = float(size_str) / 1000
        if load_str: load_hp = int(load_str)
        if rpm_str: rpm = int(rpm_str)
    return fault_type, fault_size, load_hp, rpm


# -------------------- 遍历 .mat 文件 --------------------
signals, labels = [], []
fault_map = {'N': 0, 'OR': 1, 'IR': 2, 'B': 3}

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith('.mat'):
            path = os.path.join(root, f)
            try:
                data = loadmat(path)
                key = next(k for k in data if k[0].isalpha())  # 取信号
                sig = data[key].flatten()
                signals.append(sig)
                fault_type, _, _, _ = parse_cwru_filename(os.path.splitext(f)[0])
                labels.append(fault_map.get(fault_type, -1))
            except Exception as e:
                print(f"⚠️ 跳过 {f}: {e}")

signals = np.array(signals, dtype=object)
labels = np.array(labels)
np.save(os.path.join(DATA_DIR, 'cwru_signals.npy'), signals)
np.save(os.path.join(DATA_DIR, 'cwru_labels.npy'), labels)

print(f"✅ 完成 CWRU 数据下载与解析")
print(f"总信号数: {len(signals)}, 标签分布: {np.unique(labels, return_counts=True)}")
