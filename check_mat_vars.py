#查看任意一个目标域文件的内部结构。
from scipy.io import loadmat
import os

# --- 请修改为您目标域.mat文件的实际路径 ---
# 假设您的文件名为 A.mat，并且在 data/target 文件夹下
target_file_path = os.path.join('data', 'target', 'A.mat')

try:
    # 加载 .mat 文件
    data_dict = loadmat(target_file_path)

    print(f"成功加载文件: {target_file_path}")
    print("\n文件包含以下变量 (Keys):")

    # 打印所有变量名
    for key in data_dict.keys():
        # 忽略以'__'开头的内部变量
        if not key.startswith('__'):
            variable_data = data_dict[key]
            # 打印变量名、形状和类型，帮助我们识别
            print(f"  - 变量名: '{key}', 形状: {variable_data.shape}, 类型: {type(variable_data)}")

except FileNotFoundError:
    print(f"错误: 文件未找到，请确认路径 '{target_file_path}' 是否正确。")
except Exception as e:
    print(f"加载文件时出错: {e}")