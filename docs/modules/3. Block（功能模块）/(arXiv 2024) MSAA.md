# CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2405.10530](https://arxiv.org/pdf/2405.10530)
- **源文件**: `(arXiv 2024) MSAA.py`

### 设计机制
- # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
- x_1_2_fusion = self.fusion_1x2(x1, x2)
- x_1_4_fusion = self.fusion_1x4(x1, x4)
- x_fused = x_1_2_fusion + x_1_4_fusion
- Print the shapes of the inputs and the output

## 2. 核心分析
### 类定义与参数
#### `class ChannelAttentionModule`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, reduction`

#### `class SpatialAttentionModule`
- **描述**: 无文档说明。
- **初始化参数**: `kernel_size`

#### `class FusionConv`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, factor`

#### `class MSAA`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2024) MSAA import ...

block = MSAA(in_channels=64, out_channels=128)
    x1 = torch.randn(1, 64, 64, 64)
    x2 = torch.randn(1, 64, 64, 64)
    x4 = torch.randn(1, 64, 64, 64)

    output = block(x1, x2, x4)

    # Print the shapes of the inputs and the output
    print(x1.size())
    print(x2.size())
    print(x4.size())
    print(output.size())
```

## 4. 适用任务
- **语义分割/实例分割**
- **遥感图像处理**
- **图像融合**
- **注意力机制应用**
- **长序列建模/Mamba应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
