import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os


# --- 类的定义保留在顶层 ---
class BearingDataset(Dataset):
    def __init__(self, metadata_path, data_root_path, domain_label, window_size=2048, step_size=1024):
        super(BearingDataset, self).__init__()
        self.window_size = window_size
        self.step_size = step_size
        self.domain_label = domain_label

        self.path_map = {}
        print(f"正在扫描 {data_root_path} 目录下的所有 .mat 文件...")
        for root, _, files in os.walk(data_root_path):
            for file in files:
                if file.endswith('.mat'):
                    self.path_map[file] = os.path.join(root, file)

        self.metadata = pd.read_csv(metadata_path)
        original_rows = len(self.metadata)
        self.metadata = self.metadata[self.metadata['FileName'].isin(self.path_map.keys())]
        retained_rows = len(self.metadata)
        print(f"元数据与实际文件匹配完成。共 {retained_rows} / {original_rows} 个文件被保留。")

        self.index_map = []
        print(f"正在为域 {domain_label} 构建索引...")
        for _, row in self.metadata.iterrows():
            try:
                correct_file_path = self.path_map[row['FileName']]
                mat_data = loadmat(correct_file_path)
                filename_no_ext = os.path.splitext(row['FileName'])[0]
                signal = self._get_signal_from_mat(mat_data, filename_no_ext)

                if signal is not None and len(signal) >= self.window_size:
                    num_slices = (len(signal) - self.window_size) // self.step_size + 1
                    for i in range(num_slices):
                        start_point = i * self.step_size
                        self.index_map.append((row.name, start_point))
            except Exception as e:
                print(f"警告：无法加载或处理文件 {row['FileName']}: {e}")

        print(f"域 {domain_label} 构建完成，总样本数 (分片数): {len(self.index_map)}")

    def _get_signal_from_mat(self, mat_data, filename_key):
        clean_filename_key = filename_key.split('_(')[0]
        if clean_filename_key in mat_data:
            return mat_data[clean_filename_key].flatten()
        for key in mat_data.keys():
            if 'time' in key.lower() or key.startswith('X'):
                return mat_data[key].flatten()
        return None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        row_idx, start_point = self.index_map[idx]
        file_info = self.metadata.loc[row_idx]
        file_name = file_info['FileName']
        fault_label = file_info['FaultType_Label']
        correct_file_path = self.path_map[file_name]
        mat_data = loadmat(correct_file_path)
        filename_no_ext = os.path.splitext(file_name)[0]
        signal = self._get_signal_from_mat(mat_data, filename_no_ext)
        end_point = start_point + self.window_size
        signal_slice = signal[start_point:end_point]
        signal_tensor = torch.from_numpy(signal_slice).float().unsqueeze(0)
        fault_label_tensor = torch.tensor(fault_label, dtype=torch.long)
        domain_label_tensor = torch.tensor(self.domain_label, dtype=torch.long)
        return signal_tensor, fault_label_tensor, domain_label_tensor


# ▼▼▼ 核心修改：将所有逻辑封装到一个函数中 ▼▼▼
def get_loaders(batch_size=64):
    """
    创建并返回所有需要的数据加载器
    """
    SOURCE_METADATA_PATH = 'source_data_metadata.csv'
    TARGET_METADATA_PATH = 'target_data_metadata.csv'
    SOURCE_DATA_ROOT = 'data/origin'
    TARGET_DATA_ROOT = 'data/target'

    print("\n--- 初始化源域和目标域完整数据集 ---")
    source_dataset_full = BearingDataset(SOURCE_METADATA_PATH, SOURCE_DATA_ROOT, domain_label=0)
    target_dataset = BearingDataset(TARGET_METADATA_PATH, TARGET_DATA_ROOT, domain_label=1)

    print("\n--- 正在划分源域数据集为训练集和测试集 ---")
    test_split_ratio = 0.2
    num_source_samples = len(source_dataset_full)
    indices = list(range(num_source_samples))
    split = int(np.floor(test_split_ratio * num_source_samples))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    source_train_dataset = Subset(source_dataset_full, train_indices)
    source_test_dataset = Subset(source_dataset_full, test_indices)

    print(f"源域训练集样本数: {len(source_train_dataset)}")
    print(f"源域测试集样本数: {len(source_test_dataset)}")
    print(f"目标域样本数: {len(target_dataset)}")

    # 创建数据加载器
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True)
    source_test_loader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("\n数据加载器创建成功!")
    return source_train_loader, source_test_loader, target_loader

if __name__ == '__main__':
    # --- 这个代码块仅用于直接运行此文件时进行测试 ---
    BATCH_SIZE = 64
    SOURCE_METADATA_PATH = 'source_data_metadata.csv'
    TARGET_METADATA_PATH = 'target_data_metadata.csv'
    SOURCE_DATA_ROOT = 'data/origin'
    TARGET_DATA_ROOT = 'data/target'

    print("\n--- 初始化源域数据集 ---")
    source_dataset_full = BearingDataset(SOURCE_METADATA_PATH, SOURCE_DATA_ROOT, domain_label=0)

    print("\n--- 初始化目标域数据集 ---")
    target_dataset = BearingDataset(TARGET_METADATA_PATH, TARGET_DATA_ROOT, domain_label=1)

    print("\n--- 将源域数据集划分为训练集和测试集 (用于后续评估) ---")
    test_split_ratio = 0.2
    num_source_samples = len(source_dataset_full)
    indices = list(range(num_source_samples))
    split = int(np.floor(test_split_ratio * num_source_samples))
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    source_train_dataset = Subset(source_dataset_full, train_indices)
    source_test_dataset = Subset(source_dataset_full, test_indices)

    print(f"源域训练集样本数: {len(source_train_dataset)}")
    print(f"源域测试集样本数: {len(source_test_dataset)}")

    # --- 创建数据加载器 ---
    source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                     pin_memory=True)
    source_test_loader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                    pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print("\n数据加载器创建成功!")