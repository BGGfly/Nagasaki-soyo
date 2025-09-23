import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- 参数 --------------------
SOURCE_FEATURE_FILE = 'step3_features.npy'
SOURCE_LABEL_FILE = 'step3_labels.npy'
OUTPUT_DIR = os.path.join('features', 'NN', 'task2')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
EPOCHS = 60
LEARNING_RATE = 5e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- 数据加载 --------------------
X = np.load(SOURCE_FEATURE_FILE)
y = np.load(SOURCE_LABEL_FILE)

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
np.savez(os.path.join(OUTPUT_DIR, 'feature_scaler.npz'), mean=scaler.mean_, scale=scaler.scale_)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# -------------------- 类别权重 & 采样 --------------------
class_counts = np.bincount(y)
class_weights = 1. / class_counts
sample_weights = class_weights[y]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


# -------------------- 神经网络定义 --------------------
class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


model = Net(X.shape[1], len(np.unique(y))).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------- 训练 --------------------
best_loss = float('inf')
train_loss_hist, train_acc_hist = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(dim=1) == yb).sum().item()
    epoch_loss = total_loss / len(dataset)
    epoch_acc = correct / len(dataset)
    train_loss_hist.append(epoch_loss)
    train_acc_hist.append(epoch_acc)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # 保存最优模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'task2_nn_model.pth'))

# -------------------- 可视化训练曲线 --------------------
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_loss_hist, label='Loss')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.title('Training Loss');
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'train_loss.png'));
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_acc_hist, label='Accuracy')
plt.xlabel('Epoch');
plt.ylabel('Accuracy');
plt.title('Training Accuracy');
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'train_acc.png'));
plt.close()

print(f"✅ 模型已保存到 {OUTPUT_DIR}/task2_nn_model.pth")
