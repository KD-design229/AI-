# ABC: Attention with Bilinear Correlation for Infrared Small Target Detection ICME2023

## 1. 模块简介
- **相关论文/地址**: [https://arxiv.org/pdf/2303.10321](https://arxiv.org/pdf/2303.10321)
- **源文件**: `(ICME 2023) CLFT.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `BAM`: 该模块实现的核心类之一。
- `Conv`: 该模块实现的核心类之一。
- `DConv`: 该模块实现的核心类之一。
- `ConvAttention`: 该模块实现的核心类之一。
- `FeedForward`: 该模块实现的核心类之一。
- `CLFT`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `(ICME 2023) CLFT.py` 中的代码复制到项目中，或者通过 `from (ICME 2023) CLFT import CLFT` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
