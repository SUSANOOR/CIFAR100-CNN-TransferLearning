import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import json
import os

# 导入之前定义的数据集加载器
from dataset_setup import train_loader, val_loader, test_loader


def train_transfer():
    # 检查并设置设备 (RTX 3050 Ti)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 加载预训练的 ResNet50
    base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 核心适配：将图片上采样至 128 像素以匹配预训练权重的特征提取能力 [cite: 11]
    class ResNetTransfer(nn.Module):
        def __init__(self, model):
            super(ResNetTransfer, self).__init__()
            self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
            self.model = model
            # 修改最后的全连接层以适配 100 个类别 [cite: 11]
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 100)

        def forward(self, x):
            x = self.upsample(x)
            x = self.model(x)
            return x

    model = ResNetTransfer(base_model).to(device)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)

    # 余弦退火学习率调度
    epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 混合精度训练，节省 4GB 显存
    scaler = torch.cuda.amp.GradScaler()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("开始高性能迁移学习微调...")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 恢复与基准代码一致的输出格式
        print(
            f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 测试集最终评估 [cite: 11]
    print("正在进行测试集最终评估...")
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc_raw = test_correct / test_total
    # 最终结果输出百分比
    print(f"迁移学习模型测试集最终准确率: {test_acc_raw * 100:.2f}%")
    history['test_acc'] = test_acc_raw

    # 保存权重与训练历史
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/transfer_resnet50_model.pth')
    with open('results/transfer_history.json', 'w') as f:
        json.dump(history, f)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transfer Learning Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Transfer Learning Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/transfer_curves.png')

    # 绘制与基准模型的对比图 [cite: 11]
    baseline_path = 'results/baseline_history.json'
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            base_hist = json.load(f)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(base_hist['val_loss'], '--', label='Baseline Val Loss')
        plt.plot(history['val_loss'], label='Transfer Val Loss')
        plt.title('Validation Loss Comparison')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(base_hist['val_acc'], '--', label='Baseline Val Acc')
        plt.plot(history['val_acc'], label='Transfer Val Acc')
        plt.title('Validation Accuracy Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/comparison_baseline_vs_transfer.png')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_transfer()