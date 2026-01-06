# Attention Multihop Graph and Multiscale Convolutional Fusion Network for Hyperspectral Image Classification

## 1. 模块简介
- **相关论文/地址**: [https://ieeexplore.ieee.org/document/10098209](https://ieeexplore.ieee.org/document/10098209)
- **源文件**: `(IEEE 2023) CAFM.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `MatMul`: 该模块实现的核心类之一。
- `LinAngularAttention`: 该模块实现的核心类之一。
- `XCA`: 该模块实现的核心类之一。
- `CAFM`: 该模块实现的核心类之一。
- `LinAngularXCA_CA`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `(IEEE 2023) CAFM.py` 中的代码复制到项目中，或者通过 `from (IEEE 2023) CAFM import LinAngularXCA_CA` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
