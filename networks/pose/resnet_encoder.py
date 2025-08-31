import torch
import numpy as np

from torch import nn
from torchvision import models


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    适用于多帧输入的ResNet模型，支持多个图像帧作为输入。
    该模型继承自torchversion的ResNet，实现版本与原生ResNet基本相同
    修改：第一层卷积层(conv1)以适配多帧输入
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64  # 初始化输入通道数
        self.conv1 = nn.Conv2d(  # 修改第一层卷积，使其适配num_input_images帧输入
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 保持后续层不变
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multi_image_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50  ResNet层数
        pretrained (bool): If True, returns a model pre-trained on ImageNet  是否加载预训练模型
        num_input_images (int): Number of frames stacked as input  输入的图像帧数
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock,
                  50: models.resnet.Bottleneck}[num_layers]
    # 创建模型
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        if num_layers == 18:
            model_pretrained = models.resnet18(pretrained=True)
        elif num_layers == 50:
            model_pretrained = models.resnet50(pretrained=True)

            # 修改预训练模型的输入层权重以适应多个输入图像
        pretrained_weights = model_pretrained.state_dict()
        pretrained_weights['conv1.weight'] = torch.cat(
            [pretrained_weights['conv1.weight']] * num_input_images, 1) / num_input_images

        # 将预训练权重加载到自定义模型中
        model.load_state_dict(pretrained_weights)
    return model


class ResNetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    ResNet作为深度估计任务的编码器
    可选ResNet层数：18，34，50，101，152
    是否使用预训练模型
    运行多输入通道（num_input_images>1）
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResNetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multi_image_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        """
        前向传播：提取多尺度特征
        Args:
            input_image: 形状为 (B, C, H, W) 的输入图像
        Returns:
            多尺度特征列表 [C1, C2, C3, C4, C5]
        """
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features