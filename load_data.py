import os
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

# 定义数据文件所在的根目录
data_folder = 'C:\\Users\\12394\Desktop\\数据集\\数据集\\源域数据集'  # 替换为你的数据实际路径

# --- 1. 获取所有子文件夹下的 .mat 文件 ---
# 方法一：使用 os.walk 遍历所有子文件夹 (推荐)
all_mat_files = []
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.mat'):
            all_mat_files.append(os.path.join(root, file))

print(f'检测到 {len(all_mat_files)} 个 .mat 文件。')

if all_mat_files:
    first_file_path = all_mat_files[0]
    print(f'正在加载第一个文件: {first_file_path}')

    # 加载 .mat 文件
    # loadmat 默认会加载所有变量，并返回一个字典
    data_dict = loadmat(first_file_path)

    print('文件内容变量如下：')
    # loadmat 会自动添加一些元数据，我们只关心实际数据变量
    for key in data_dict.keys():
        if not key.startswith('__'): # 忽略Python内部变量
            print(f'- {key}')

    # 尝试显示其中一个振动信号的数据长度和部分值
    # 根据附件描述，变量可能命名为 X***_DE_time, X***_FE_time 等
    # 遍历字典的键，找到包含 'DE_time' 或 'FE_time' 的时间序列数据
    found_signal = False
    for key in data_dict.keys():
        if ('DE_time' in key or 'FE_time' in key) and isinstance(data_dict[key], np.ndarray) and data_dict[key].ndim == 2:
            print(f'\n找到了振动信号变量: {key}')
            signal_data = data_dict[key].flatten() # 将二维数组展平为一维
            print(f'信号数据长度: {len(signal_data)}')
            print(f'信号数据前10个点: \n{signal_data[:min(10, len(signal_data))]}')
            found_signal = True
            break

    if not found_signal:
        print('\n未找到符合DE_time或FE_time模式的振动信号变量。请手动检查文件内容。')

    # 尝试显示RPM
    found_rpm = False
    for key in data_dict.keys():
        if 'RPM' in key and isinstance(data_dict[key], np.ndarray) and data_dict[key].size == 1:
            print(f'\n找到了转速变量: {key}')
            print(f'转速 (RPM): {data_dict[key].item():.2f}') # .item() 用于从 numpy 标量数组中取值
            found_rpm = True
            break

    if not found_rpm:
        print('\n未找到符合RPM模式的转速变量。')

else:
    print('未找到任何 .mat 文件，请检查路径是否正确或子文件夹中是否包含文件。')