# HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2403.10778](https://arxiv.org/pdf/2403.10778)
- **源文件**: `(arXiv2024) PPA.py`

### 设计机制
- 中文题目: HCF-Net:用于红外小目标检测的层次化上下文融合网络
- 官方github：https://github.com/zhengshuchen/HCFNet
- Local branch

## 2. 核心分析
### 类定义与参数
#### `class SpatialAttentionModule`
- **描述**: 无文档说明。

#### `class PPA`
- **描述**: 无文档说明。
- **初始化参数**: `in_features, filters`

#### `class LocalGlobalAttention`
- **描述**: 无文档说明。
- **初始化参数**: `output_dim, patch_size`

#### `class ECA`
- **描述**: 无文档说明。
- **初始化参数**: `in_channel, gamma, b`

#### `class conv_block`
- **描述**: 无文档说明。
- **初始化参数**: `in_features, out_features, kernel_size, stride, padding, dilation, norm_type, activation, use_bias, groups`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv2024) PPA import ...

# 输入通道数，输出通道数
    ppa = PPA(in_features=64, filters=64) 
    input = torch.rand(1, 64, 128, 128)  # 输入 B C H W
    output = ppa(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
```

## 4. 适用任务
- **目标检测**
- **图像融合**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
