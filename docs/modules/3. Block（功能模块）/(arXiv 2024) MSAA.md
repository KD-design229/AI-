# CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation

## 1. 模块简介
- **相关论文/地址**: [https://arxiv.org/pdf/2405.10530](https://arxiv.org/pdf/2405.10530)
- **源文件**: `(arXiv 2024) MSAA.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `ChannelAttentionModule`: 该模块实现的核心类之一。
- `SpatialAttentionModule`: 该模块实现的核心类之一。
- `FusionConv`: 该模块实现的核心类之一。
- `MSAA`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `(arXiv 2024) MSAA.py` 中的代码复制到项目中，或者通过 `from (arXiv 2024) MSAA import MSAA` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
