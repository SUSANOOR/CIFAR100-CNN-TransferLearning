import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import json
import os

# 导入数据集
from dataset_setup import train_loader, val_loader, test_loader


# 定义全新的改进模型 2 (加深网络 + 余弦退火学习率)
class ImprovedCNN2(nn.Module):
    def __init__(self, num_classes=100):
        super(ImprovedCNN2, self).__init__()
        # 改进 1：加深网络结构，采用类似 VGG 的双卷积块设计
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_and_compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    model = ImprovedCNN2(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    # 改进 2：使用余弦退火学习率调度器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("开始训练新版改进模型 2 (Deeper Network + Cosine Annealing LR)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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

        # 步进更新学习率
        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

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

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("开始在测试集上评估新版改进模型 2...")
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total
    print(f"新版改进模型 2 测试集最终准确率: {test_acc * 100:.2f}%")
    history['test_acc'] = test_acc

    # 保存权重与历史
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/improve2_model.pth')
    with open('results/improve2_history.json', 'w') as f:
        json.dump(history, f)
    print("新版改进模型 2 的权重和训练历史已保存。")

    # 绘制模型自身曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Improvement 2 (Deep): Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Improvement 2 (Deep): Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/improve2_curves.png')

    # 绘制与基准模型的对比图
    baseline_history_path = 'results/baseline_history.json'
    if os.path.exists(baseline_history_path):
        with open(baseline_history_path, 'r') as f:
            baseline_history = json.load(f)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(baseline_history['val_loss'], label='Baseline Val Loss', linestyle='--')
        plt.plot(history['val_loss'], label='Improve2 Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Comparison')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(baseline_history['val_acc'], label='Baseline Val Acc', linestyle='--')
        plt.plot(history['val_acc'], label='Improve2 Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Comparison')
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/comparison_baseline_vs_improve2.png')
        print("对比图已保存为 'results/comparison_baseline_vs_improve2.png'。")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_and_compare()