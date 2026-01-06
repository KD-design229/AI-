# Reciprocal Attention Mixing Transformer for Lightweight Image Restoration(CVPR 2024 Workshop)

## 1. 模块简介
- **相关论文/地址**: [https://github.com/rami0205/RAMiT](https://github.com/rami0205/RAMiT)
- **源文件**: `(CVPR 2024) RAMiT.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `DropPath`: 该模块实现的核心类之一。
- `QKVProjection`: 该模块实现的核心类之一。
- `SpatialSelfAttention`: 该模块实现的核心类之一。
- `ChannelSelfAttention`: 该模块实现的核心类之一。
- `ReshapeLayerNorm`: 该模块实现的核心类之一。
- `MobiVari1`: 该模块实现的核心类之一。
- `MobiVari2`: 该模块实现的核心类之一。
- `FeedForward`: 该模块实现的核心类之一。
- `NoLayer`: 该模块实现的核心类之一。
- `DRAMiTransformer`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `(CVPR 2024) RAMiT.py` 中的代码复制到项目中，或者通过 `from (CVPR 2024) RAMiT import DRAMiTransformer` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
