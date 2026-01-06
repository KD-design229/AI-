# Fast_Fourier_Convolution

## 1. 模块简介
- **源文件**: `Fast_Fourier_Convolution.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `FourierUnit`: 该模块实现的核心类之一。
- `SpectralTransform`: 该模块实现的核心类之一。
- `FFC`: 该模块实现的核心类之一。
- `FFC_BN_ACT`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `Fast_Fourier_Convolution.py` 中的代码复制到项目中，或者通过 `from Fast_Fourier_Convolution import FFC_BN_ACT` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
