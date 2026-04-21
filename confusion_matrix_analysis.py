import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# 根据你的项目文件名进行导入
from dataset_setup import test_loader, test_dataset
from base_cnn import BaselineCNN
from train_improve1 import ImprovedCNN1
from train_improve2 import ImprovedCNN2


# 定义迁移学习模型结构
class ResNetTransfer(nn.Module):
    def __init__(self, model):
        super(ResNetTransfer, self).__init__()
        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.model = model
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 100)

    def forward(self, x):
        x = self.upsample(x)
        x = self.model(x)
        return x


def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds


def plot_confusion_matrix(model_name, labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(20, 18))
    # 对于 100 类，不显示具体数值(annot=False)，否则会重叠
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/confusion_matrix_{model_name.lower()}.png', dpi=300)
    print(f"{model_name} 的混淆矩阵已保存至 results 文件夹。")
    plt.close()


def run_analysis():
    # 使用 CPU 进行推理，确保 4GB 显存不会溢出
    device = torch.device("cpu")
    class_names = test_dataset.classes

    model_configs = [
        {'name': 'Baseline', 'class': BaselineCNN, 'path': 'results/baseline_model.pth'},
        {'name': 'Improve1', 'class': ImprovedCNN1, 'path': 'results/improve1_model.pth'},
        {'name': 'Improve2', 'class': ImprovedCNN2, 'path': 'results/improve2_model.pth'},
        {'name': 'ResNet50', 'class': lambda: ResNetTransfer(models.resnet50()),
         'path': 'results/transfer_resnet50_model.pth'}
    ]

    for cfg in model_configs:
        if not os.path.exists(cfg['path']):
            print(f"跳过 {cfg['name']}，未找到权重文件。")
            continue

        print(f"正在计算 {cfg['name']} 的混淆矩阵数据...")
        model = cfg['class']().to(device)
        model.load_state_dict(torch.load(cfg['path'], map_location=device, weights_only=True))

        labels, preds = get_predictions(model, test_loader, device)
        plot_confusion_matrix(cfg['name'], labels, preds, class_names)


if __name__ == '__main__':
    run_analysis()