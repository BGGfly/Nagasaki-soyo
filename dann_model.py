import torch
import torch.nn as nn
from torch.autograd import Function


# --- 1. 定义梯度反转层 (GRL) ---
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 将接收到的梯度乘以-lambda_val再传回去
        return (grad_output.neg() * ctx.lambda_val), None


def grad_reverse(x, lambda_val=1.0):
    return GradientReversalLayer.apply(x, lambda_val)


# --- 2. 定义特征提取器 (1D-CNN) ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入形状: (batch, 1, 2048)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 64)
        return x


# --- 3. 定义通用分类器 (MLP) ---
class Classifier(nn.Module):
    def __init__(self, input_features=64, hidden_dim=100, num_classes=4):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


# --- 4. 组合成完整的DANN模型 ---
class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        # 标签分类器：输出4个类别 (N, OR, IR, B)
        self.label_classifier = Classifier(input_features=64, num_classes=4)
        # 域分类器：输出2个类别 (源域 vs 目标域)
        self.domain_classifier = Classifier(input_features=64, num_classes=2)

    def forward(self, x, lambda_val=1.0):
        features = self.feature_extractor(x)
        label_output = self.label_classifier(features)

        # 在域分类路径上应用梯度反转
        reversed_features = grad_reverse(features, lambda_val)
        domain_output = self.domain_classifier(reversed_features)

        return label_output, domain_output


# --- 5. 测试模型定义是否正确的代码块 ---
if __name__ == '__main__':
    BATCH_SIZE = 64
    print("--- 测试DANN模型定义 ---")
    dummy_input = torch.randn(BATCH_SIZE, 1, 2048)
    model = DANN()
    label_out, domain_out = model(dummy_input)

    print("模型实例化成功！")
    print(f"模拟输入形状: {dummy_input.shape}")
    print(f"标签分类器输出形状: {label_out.shape}")  # 应该为 (64, 4)
    print(f"域分类器输出形状: {domain_out.shape}")  # 应该为 (64, 2)