import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

try:
    from dann_data_loader import get_loaders
    from dann_model import DANN
except ImportError:
    print("错误：无法导入'dann_data_loader.py'或'dann_model.py'。")
    exit()


def main():
    # --- 1. 定义超参数 ---
    LEARNING_RATE = 0.001
    ALPHA = 0.1  # 对抗权重
    NUM_EPOCHS = 100
    WARMUP_EPOCHS = 10  # 新增：前10轮只做分类训练，不进行对抗

    # --- 2. 初始化 ---
    print("--- 步骤1: 初始化 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    source_train_loader, source_test_loader, target_loader = get_loaders()
    model = DANN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    history = {'epoch': [], 'source_accuracy': []}
    best_accuracy = 0.0

    print(f"\n--- 步骤2: 开始进行 {NUM_EPOCHS} 轮的训练 ---")
    print(f"注意：前 {WARMUP_EPOCHS} 轮为热身阶段，仅训练分类器。")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        len_source_loader = len(source_train_loader)
        len_target_loader = len(target_loader)
        max_num_iter = max(len_source_loader, len_target_loader)
        iter_source = iter(source_train_loader)
        iter_target = iter(target_loader)

        # ▼▼▼ 核心修改：引入热身逻辑 ▼▼▼
        # 根据当前epoch决定是否开启对抗
        is_adversarial = (epoch > WARMUP_EPOCHS)
        current_alpha = ALPHA if is_adversarial else 0  # 热身阶段，对抗权重为0

        desc = f"Epoch {epoch}/{NUM_EPOCHS} [对抗开启]" if is_adversarial else f"Epoch {epoch}/{NUM_EPOCHS} [热身]"

        for i in tqdm(range(max_num_iter), desc=desc):
            # ... (数据加载逻辑不变) ...
            try:
                source_data, source_label, _ = next(iter_source)
            except StopIteration:
                iter_source = iter(source_train_loader)
                source_data, source_label, _ = next(iter_source)
            try:
                target_data, _, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(target_loader)
                target_data, _, _ = next(iter_target)
            if source_data.size(0) != target_data.size(0):
                min_size = min(source_data.size(0), target_data.size(0))
                source_data, source_label = source_data[:min_size], source_label[:min_size]
                target_data = target_data[:min_size]
            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                p = float(i + (epoch - 1) * max_num_iter) / (NUM_EPOCHS * max_num_iter)
                lambda_val = 2. / (1. + np.exp(-10 * p)) - 1

                # 标签分类损失 (始终计算)
                label_output, domain_output_s = model(source_data, lambda_val=lambda_val)
                err_s_label = loss_class(label_output, source_label)

                # 域损失 (只在对抗阶段计算)
                if is_adversarial:
                    domain_label_source = torch.zeros(source_data.size(0)).long().to(device)
                    err_s_domain = loss_domain(domain_output_s, domain_label_source)
                    _, domain_output_t = model(target_data, lambda_val=lambda_val)
                    domain_label_target = torch.ones(target_data.size(0)).long().to(device)
                    err_t_domain = loss_domain(domain_output_t, domain_label_target)
                    err_domain = err_s_domain + err_t_domain
                else:
                    err_domain = torch.tensor(0.0).to(device)  # 热身阶段，域损失为0

                # 使用 current_alpha 控制对抗权重
                err_total = err_s_label + current_alpha * err_domain

            scaler.scale(err_total).backward()
            scaler.step(optimizer)
            scaler.update()

        # --- 评估部分 ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels, _ in source_test_loader:
                data, labels = data.to(device), labels.to(device)
                label_output, _ = model(data, lambda_val=0)
                _, predicted = torch.max(label_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        history['epoch'].append(epoch)
        history['source_accuracy'].append(accuracy)

        print(f'\nEpoch [{epoch}/{NUM_EPOCHS}], LR: {scheduler.get_last_lr()[0]:.6f}, Source Test Acc: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'dann_model_best.pth')
            print(f"检测到新的最佳准确率: {best_accuracy:.2f}%。模型已保存。")

        scheduler.step()

    print("\n--- 训练完成 ---")
    print(f"训练期间在源域测试集上达到的最佳准确率为: {best_accuracy:.2f}%")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()