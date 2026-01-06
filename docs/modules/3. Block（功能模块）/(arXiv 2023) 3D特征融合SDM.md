# PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion

## 1. 模块简介
- **源文件**: `(arXiv 2023) 3D特征融合SDM.py`

### 设计机制
- 3D图像分割即插即用模块
- self.conv1 = Conv3dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
- initialize

## 2. 核心分析
### 类定义与参数
#### `class SDC`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, guidance_channels, kernel_size, stride, padding, dilation, groups, bias, theta`

#### `class SDM`
- **描述**: 无文档说明。
- **初始化参数**: `in_channel, guidance_channels`

#### `class Conv3dReLU`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, kernel_size, padding, stride, use_batchnorm`

#### `class Conv3dbn`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, kernel_size, padding, stride, use_batchnorm`

#### `class Conv3dGNReLU`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, kernel_size, padding, stride, use_batchnorm`

#### `class Conv3dGN`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, kernel_size, padding, stride, use_batchnorm`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2023) 3D特征融合SDM import ...

import torch

    # 定义输入张量的形状
    input = (1, 3, 32, 32, 32)  # 输入 B C D H W

    # 创建输入张量
    input_tensor = torch.randn(input)

    # 创建引导张量
    guidance_tensor = torch.randn((1, 2, 32, 32, 32))  # 假设引导张量与输入张量大小相同

    # 创建模型
    block = SDM(in_channel=3, guidance_channels=2)

    # 将模型设置为评估模式
    block.eval()

    # 打印输入张量的形状
    print(input_tensor.size())

    # 执行前向传播
    output_tensor = block(input_tensor, guidance_tensor)

    # 打印输出张量的形状
    print(output_tensor.size())
```

## 4. 适用任务
- **语义分割/实例分割**
- **图像融合**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
