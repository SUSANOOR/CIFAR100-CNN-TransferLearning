import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os

# 导入数据集加载器
from dataset_setup import test_loader, test_dataset


# 1. 定义与训练时完全一致的模型结构
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


def perform_error_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    # 2. 加载模型与权重
    base_model = models.resnet50()
    model = ResNetTransfer(base_model).to(device)

    model_path = 'results/transfer_resnet50_model.pth'
    if not os.path.exists(model_path):
        print(f"错误：未找到模型权重文件 {model_path}，请先运行 train_transfer.py")
        return

    # 加载权重 (使用 weights_only=True 保证安全)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 获取类别名称
    class_names = test_dataset.classes

    # 3. 寻找错误样本
    wrong_samples = []

    print("正在测试集上检索错误样本...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 找出预测错误的索引
            incorrect_mask = preds != labels
            if incorrect_mask.any():
                for i in range(len(inputs)):
                    if incorrect_mask[i]:
                        # 记录图片（转回CPU）、预测值、真实值
                        wrong_samples.append({
                            'img': inputs[i].cpu(),
                            'pred': preds[i].item(),
                            'label': labels[i].item()
                        })

                    if len(wrong_samples) >= 5:
                        break
            if len(wrong_samples) >= 5:
                break

    # 4. 可视化分析
    plt.figure(figsize=(15, 6))

    # CIFAR-100 归一化逆操作
    mean = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    for i, sample in enumerate(wrong_samples):
        img = sample['img'].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        title = f"True: {class_names[sample['label']]}\nPred: {class_names[sample['pred']]}"
        plt.title(title, color='red', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/error_analysis.png')
    print("错误样本分析图已保存至 'results/error_analysis.png'")
    plt.show()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    perform_error_analysis()