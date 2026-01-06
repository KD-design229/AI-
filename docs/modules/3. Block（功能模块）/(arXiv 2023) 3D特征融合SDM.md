# PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion

## 1. 模块简介
- **源文件**: `(arXiv 2023) 3D特征融合SDM.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `SDC`: 该模块实现的核心类之一。
- `SDM`: 该模块实现的核心类之一。
- `Conv3dReLU`: 该模块实现的核心类之一。
- `Conv3dbn`: 该模块实现的核心类之一。
- `Conv3dGNReLU`: 该模块实现的核心类之一。
- `Conv3dGN`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `(arXiv 2023) 3D特征融合SDM.py` 中的代码复制到项目中，或者通过 `from (arXiv 2023) 3D特征融合SDM import Conv3dGN` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
