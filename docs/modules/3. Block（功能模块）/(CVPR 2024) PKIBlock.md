# Poly Kernel Inception Network for Remote Sensing Detection(CVPR 2024)

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2403.06258](https://arxiv.org/pdf/2403.06258)
- **源文件**: `(CVPR 2024) PKIBlock.py`

### 设计机制
- Poly Kernel Inception Block(PKIBlock)
- Add more normalization types if needed
- Add more activation types if needed
- Update InceptionBottleneck's constructor call to avoid conflicts

## 2. 核心分析
### 类定义与参数
#### `class ConvModule`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, kernel_size, stride, padding, dilation, groups, norm_cfg, act_cfg`

#### `class InceptionBottleneck`
- **描述**: Bottleneck with Inception module
- **初始化参数**: `in_channels, out_channels, kernel_sizes, dilations, expansion, add_identity, with_caa, caa_kernel_size, norm_cfg, act_cfg`

#### `class CAA`
- **描述**: Context Anchor Attention
- **初始化参数**: `channels, h_kernel_size, v_kernel_size, norm_cfg, act_cfg`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2024) PKIBlock import ...

input = torch.randn(1, 64, 128, 128) #输入B C H W
    block = InceptionBottleneck(in_channels=64, out_channels=128)
    output = block(input)
    print(input.size())
    print(output.size())
```

## 4. 适用任务
- **目标检测**
- **图像分类**
- **图像去噪**
- **遥感图像处理**
- **目标跟踪**
- **去雨**
- **去雾**
- **去模糊**
- **图像融合**
- **语义分割/实例分割**
- **超分辨率**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
