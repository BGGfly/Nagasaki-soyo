import os
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt

# -------------------- 参数 --------------------
FEATURE_DIM = 13
NUM_CLASSES = 4
TARGET_DIR = "data/target"  # 16 个 A.mat~P.mat
MODEL_PATH = "features/NN/task3/task3_nn_model.pth"
OUTPUT_DIR = "features/NN/task4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- 模型定义 --------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return x

# -------------------- 加载模型 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

# 使用 strict=False，兼容 Task3 模型和当前模型结构差异
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ 已加载 Task3NN 模型")

# -------------------- 特征提取函数 --------------------
def extract_features(signal, fs=12000):
    window_size = 2048
    step_size = 1024
    feats_list = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        seg = signal[start:start + window_size]
        feats = [
            np.mean(seg),
            np.std(seg),
            np.max(seg),
            np.min(seg),
            np.ptp(seg),
            np.median(seg),
            np.mean(np.abs(seg)),
            np.sqrt(np.mean(seg**2)),
            np.percentile(seg, 25),
            np.percentile(seg, 75),
            np.sum(seg**2),
            np.sum(np.abs(seg)),
            np.mean(np.diff(seg)**2)
        ]
        feats_list.append(feats)

    feats_array = np.mean(feats_list, axis=0)
    return np.array(feats_array, dtype=np.float32)

def extract_features_from_mat(file_path):
    data_dict = loadmat(file_path)
    var_name = os.path.splitext(os.path.basename(file_path))[0]
    if var_name not in data_dict:
        raise ValueError(f"{file_path} 中没有找到变量 {var_name}")
    signal = data_dict[var_name].flatten()
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    feats = extract_features(signal)
    return feats

# -------------------- 预测 --------------------
file_list = sorted([f for f in os.listdir(TARGET_DIR) if f.endswith(".mat")])
pred_probs = []
pred_labels = []

for f in file_list:
    file_path = os.path.join(TARGET_DIR, f)
    feats = extract_features_from_mat(file_path)
    feats_tensor = torch.from_numpy(feats).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(feats_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        label = np.argmax(probs)

    pred_probs.append(probs)
    pred_labels.append(label)
    print(f"{f} --> {['N', 'IR', 'OR', 'B'][label]} (prob: {probs})")

pred_probs = np.array(pred_probs)
pred_labels = np.array(pred_labels)

# -------------------- 保存结果 --------------------
np.save(os.path.join(OUTPUT_DIR, "target_pred_probs.npy"), pred_probs)
np.save(os.path.join(OUTPUT_DIR, "target_pred_labels.npy"), pred_labels)

# -------------------- 可视化柱状图 --------------------
plt.figure(figsize=(14, 6))
states = ['N', 'IR', 'OR', 'B']
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
x = np.arange(len(file_list))
width = 0.2

for i, (s, c) in enumerate(zip(states, colors)):
    bars = plt.bar(x + i*width, pred_probs[:, i], width=width, label=s, color=c)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}",
                 ha='center', va='bottom', fontsize=9)

plt.xticks(x + 1.5*width, file_list, rotation=45, ha='right')
plt.ylabel("Probability")
plt.title("Target Domain Prediction Probabilities")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "target_pred_barplot_enhanced.png"))
plt.show()

print(f"✅ Task4 NN 预测完成，结果保存在 {OUTPUT_DIR}/")
