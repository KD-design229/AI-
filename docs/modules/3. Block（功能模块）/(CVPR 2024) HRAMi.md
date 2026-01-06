# Reciprocal Attention Mixing Transformer for Lightweight Image Restoration(CVPR 2024 Workshop)

## 1. 模块简介
- **论文地址**: [https://arxiv.org/abs/2305.11474](https://arxiv.org/abs/2305.11474)
- **源文件**: `(CVPR 2024) HRAMi.py`

### 设计机制
- H-RAMi(Hierarchical Reciprocal Attention Mixer)
- Create sample input tensors
- Assume the input tensors have spatial dimensions of 32x32, 16x16, 8x8, etc.
- Pass the input through HRAMi
- Print the shapes of input and output

## 2. 核心分析
### 类定义与参数
#### `class MobiVari1`
- **描述**: 无文档说明。
- **初始化参数**: `dim, kernel_size, stride, act, out_dim`

#### `class MobiVari2`
- **描述**: 无文档说明。
- **初始化参数**: `dim, kernel_size, stride, act, out_dim, exp_factor, expand_groups`

#### `class HRAMi`
- **描述**: 无文档说明。
- **初始化参数**: `dim, kernel_size, stride, mv_ver, mv_act, exp_factor, expand_groups`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2024) HRAMi import ...

hrami = HRAMi(dim=64)

    # Create sample input tensors
    # Assume the input tensors have spatial dimensions of 32x32, 16x16, 8x8, etc.
    input = [
        torch.randn(1, 64, 32, 32),  # Level 0
        torch.randn(1, 64, 16, 16),  # Level 1
        torch.randn(1, 64, 8, 8),  # Level 2
        torch.randn(1, 64, 32, 32)  # Level 3 (final level)
    ]

    # Pass the input through HRAMi
    output = hrami(input)

    # Print the shapes of input and output
    print(f"Input shapes: {[attn.shape for attn in input]}")
    print(output.size())
```

## 4. 适用任务
- **目标检测**
- **图像分类**
- **图像去噪**
- **目标跟踪**
- **去雨**
- **去雾**
- **去模糊**
- **图像融合**
- **语义分割/实例分割**
- **超分辨率**
- **Transformer相关任务**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
