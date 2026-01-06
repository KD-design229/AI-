# BCN: Batch Channel Normalization for Image Classification

## 1. 模块简介
- **相关论文/地址**: [https://arxiv.org/pdf/2312.00596](https://arxiv.org/pdf/2312.00596)
- **源文件**: `(arXiv 2023) BCN.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `BatchNorm2D`: 该模块实现的核心类之一。
- `BatchNormm2D`: 该模块实现的核心类之一。
- `BatchNormm2DViiT`: 该模块实现的核心类之一。
- `BatchNormm2DViTC`: 该模块实现的核心类之一。
- `InstanceNorm2D`: 该模块实现的核心类之一。
- `LayerNormViT`: 该模块实现的核心类之一。
- `LayerNormViTC`: 该模块实现的核心类之一。
- `LayerNorm2D`: 该模块实现的核心类之一。
- `LayerNormm2D`: 该模块实现的核心类之一。
- `GroupNorm2D`: 该模块实现的核心类之一。
- `BatchNorm_ByoL`: 该模块实现的核心类之一。
- `LaychNorm_ByoL`: 该模块实现的核心类之一。
- `BatchNorm_Byol`: 该模块实现的核心类之一。
- `LaychNorm_Byol`: 该模块实现的核心类之一。
- `BatchChannelNorm_Byol`: 该模块实现的核心类之一。
- `BatchChannelNorm`: 该模块实现的核心类之一。
- `BatchChannelNormvit`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `(arXiv 2023) BCN.py` 中的代码复制到项目中，或者通过 `from (arXiv 2023) BCN import BatchChannelNormvit` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
