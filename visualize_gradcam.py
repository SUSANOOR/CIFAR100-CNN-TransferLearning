import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 根据你的项目文件名进行导入
from dataset_setup import test_dataset
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


# 核心修复函数：禁用所有层的 inplace 操作以兼容 Grad-CAM
def disable_inplace(model):
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # 使用 register_full_backward_hook 获取梯度
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        cam = cv2.resize(cam, (32, 32))
        return cam, class_idx


def visualize_all_gradcam():
    # 强制使用 CPU 进行可视化，避开所有显存和底层算子冲突
    device = torch.device("cpu")
    print("正在准备可视化，已自动禁用 inplace 操作以兼容梯度计算...")

    sample_idx = 10
    image_tensor, label = test_dataset[sample_idx]
    input_tensor = image_tensor.unsqueeze(0).to(device)
    class_names = test_dataset.classes

    mean, std = np.array([0.5071, 0.4867, 0.4408]), np.array([0.2675, 0.2565, 0.2761])
    img_show = np.clip(std * image_tensor.numpy().transpose((1, 2, 0)) + mean, 0, 1)

    model_configs = [
        {'name': 'Baseline', 'class': BaselineCNN, 'path': 'results/baseline_model.pth',
         'layer': lambda m: m.features[6]},
        {'name': 'Improve1', 'class': ImprovedCNN1, 'path': 'results/improve1_model.pth',
         'layer': lambda m: m.features[8]},
        {'name': 'Improve2', 'class': ImprovedCNN2, 'path': 'results/improve2_model.pth',
         'layer': lambda m: m.features[15]},
        {'name': 'ResNet50', 'class': lambda: ResNetTransfer(models.resnet50()),
         'path': 'results/transfer_resnet50_model.pth', 'layer': lambda m: m.model.layer4[-1]}
    ]

    results = []
    for cfg in model_configs:
        if not os.path.exists(cfg['path']):
            print(f"找不到权重文件: {cfg['path']}，已跳过")
            continue

        print(f"正在加载并处理: {cfg['name']}...")
        net = cfg['class']().to(device)
        net.load_state_dict(torch.load(cfg['path'], map_location=device, weights_only=True))

        # 关键步骤：在模型推理前禁用 inplace 属性
        disable_inplace(net)

        net.eval()
        cam_extractor = GradCAM(net, cfg['layer'](net))
        heatmap, pred_idx = cam_extractor.generate(input_tensor)

        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = (heatmap_color[:, :, ::-1]).astype(float) / 255
        combined = heatmap_color * 0.4 + img_show * 0.6
        results.append({'name': cfg['name'], 'img': np.clip(combined, 0, 1), 'pred': class_names[pred_idx]})

    if not results:
        print("没有可用的模型结果进行绘图。")
        return

    plt.figure(figsize=(20, 5))
    plt.subplot(1, len(results) + 1, 1)
    plt.imshow(img_show)
    plt.title(f"Original\nLabel: {class_names[label]}")
    plt.axis('off')

    for i, res in enumerate(results):
        plt.subplot(1, len(results) + 1, i + 2)
        plt.imshow(res['img'])
        plt.title(f"{res['name']}\nPred: {res['pred']}")
        plt.axis('off')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/gradcam_all_models.png')
    print("可视化完成，对比图已保存至 results/gradcam_all_models.png")
    plt.show()


if __name__ == '__main__':
    visualize_all_gradcam()