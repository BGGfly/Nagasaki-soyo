import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- 参数 --------------------
SOURCE_MODEL = 'features/NN/task2/task2_nn_model.pth'
CWRU_FEATURE_FILE = 'stepX_CWRU_features.npy'
CWRU_LABEL_FILE   = 'stepX_CWRU_labels.npy'
OUTPUT_DIR = 'features/NN/task3'
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- NN 定义（与 Task2 相同结构） --------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------- 数据加载 --------------------
X_cwru = np.load(CWRU_FEATURE_FILE)
y_cwru = np.load(CWRU_LABEL_FILE)

# 转为 Tensor
X_cwru = torch.tensor(X_cwru, dtype=torch.float32)
y_cwru = torch.tensor(y_cwru, dtype=torch.long)

# 划分 80% 微调 / 20% 测试
X_train, X_test, y_train, y_test = train_test_split(
    X_cwru, y_cwru, test_size=0.9, random_state=42, stratify=y_cwru
)

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------- 模型初始化 --------------------
input_dim = X_cwru.shape[1]
num_classes = len(np.unique(y_cwru))
model = SimpleNN(input_dim, num_classes)
model.load_state_dict(torch.load(SOURCE_MODEL))  # 加载 Task2 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------- 训练 --------------------
best_loss = float('inf')
patience_counter = 0

train_losses = []

for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss /= total
    acc = correct / total
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {acc:.4f}')

    # 早停
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'task3_nn_model.pth'))
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

# -------------------- 测试 --------------------
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'task3_nn_model.pth')))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("CWRU 测试集分类报告:")
print(classification_report(all_labels, all_preds, digits=4))

# -------------------- 混淆矩阵可视化 --------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (CWRU Test)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# -------------------- 保存训练曲线 --------------------
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'train_loss_curve.png'))
plt.close()

print(f"✅ Task3 NN 迁移学习完成，结果保存在 {OUTPUT_DIR}/")
