# DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention

## 1. 模块简介
- **相关论文/地址**: [https://github.com/cecret3350/DEA-Net/tree/main](https://github.com/cecret3350/DEA-Net/tree/main)
- **源文件**: `(TIP2024) CGAFusion.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `SpatialAttention`: 该模块实现的核心类之一。
- `ChannelAttention`: 该模块实现的核心类之一。
- `PixelAttention`: 该模块实现的核心类之一。
- `CGAFusion`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `(TIP2024) CGAFusion.py` 中的代码复制到项目中，或者通过 `from (TIP2024) CGAFusion import CGAFusion` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
