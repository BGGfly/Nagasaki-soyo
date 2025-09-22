#准备目标域数据
#先为目标域的16个.mat文件创建一个元数据CSV，16个目标域文件（A.mat, B.mat, ..., P.mat）都存放在一个名为 data/target 的文件夹里。
import os
import pandas as pd

# --- 1. 定义目标域数据所在的文件夹 ---
target_data_folder = os.path.join('data', 'target')

# 检查路径是否存在
if not os.path.exists(target_data_folder):
    print(f"错误: 目标域数据文件夹 '{target_data_folder}' 不存在。")
    print("请将您的16个目标域.mat文件（A.mat, B.mat等）放入该文件夹中。")
    exit()

# --- 2. 扫描文件并创建元数据列表 ---
all_target_files_info = []

# os.listdir() 可以获取文件夹下所有文件名
try:
    file_list = sorted(os.listdir(target_data_folder))
except FileNotFoundError:
    file_list = []

for filename in file_list:
    if filename.endswith('.mat'):
        file_path = os.path.join(target_data_folder, filename)

        # 目标域的标签是未知的，我们用 -1 或 NaN 来表示
        # 其他信息如转速(约600rpm)和采样频率(32kHz)是已知的
        all_target_files_info.append({
            'FilePath': file_path,
            'FileName': filename,
            'RPM': 600,  # 根据题目描述，转速约600rpm
            'SamplingFreq_Hz': 32000,  # 根据题目描述，采样频率32kHz
            'FaultType_Label': -1  # 使用-1表示未知标签
        })

if not all_target_files_info:
    print("警告：在目标域文件夹中没有找到任何.mat文件。")
else:
    # --- 3. 转换为DataFrame并保存 ---
    target_metadata_df = pd.DataFrame(all_target_files_info)

    output_csv_path = 'target_data_metadata.csv'
    target_metadata_df.to_csv(output_csv_path, index=False)

    print(f"成功扫描到 {len(target_metadata_df)} 个目标域文件。")
    print(f"目标域元数据已保存至 '{output_csv_path}'")
    print("\n数据预览:")
    print(target_metadata_df.head())