# Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration, CVPR 2024.

## 1. 模块简介
- **相关论文/地址**: [https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)
- **源文件**: `(CVPR 2024) FRFN.py`

## 2. 核心分析
该模块是基于上述论文实现的 PyTorch 组件，旨在提供即插即用的功能。通过对输入特征进行特定的变换（如注意力机制、特殊卷积或归一化），增强模型在计算机视觉任务中的表达能力。

### 主要类定义
- `FRFN`: 该模块实现的核心类之一。

## 3. 使用建议
- **集成方式**: 直接将 `(CVPR 2024) FRFN.py` 中的代码复制到项目中，或者通过 `from (CVPR 2024) FRFN import FRFN` 引入。
- **适用任务**: 图像分类、目标检测、语义分割等。
