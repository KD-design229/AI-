# Attention Multihop Graph and Multiscale Convolutional Fusion Network for Hyperspectral Image Classification

## 1. 模块简介
- **论文地址**: [https://ieeexplore.ieee.org/document/10098209](https://ieeexplore.ieee.org/document/10098209)
- **源文件**: `(IEEE 2023) CAFM.py`

## 2. 核心分析
### 类定义与参数
#### `class MatMul`
- **描述**: 无文档说明。

#### `class LinAngularAttention`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, num_heads, qkv_bias, attn_drop, proj_drop, res_kernel_size, sparse_reg`

#### `class XCA`
- **描述**: Cross-Covariance Attention (XCA)
Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
normalized) Cross-covariance matrix (Q^T \cdot K \in d_h \times d_h)
- **初始化参数**: `dim, num_heads, qkv_bias, attn_drop, proj_drop`

#### `class CAFM`
- **描述**: 无文档说明。

#### `class LinAngularXCA_CA`
- **描述**: 无文档说明。

## 3. 使用示例
```python
# 导入方式（参考）：from (IEEE 2023) CAFM import ...

block = LinAngularXCA_CA()
    input = torch.rand(32, 784, 128)
    output = block(input)  # 获取两个输入的输出

    print(f'Output1 shape: {output.shape}')
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
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
