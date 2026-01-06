# Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation[ECCV 2024]

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2405.06228](https://arxiv.org/pdf/2405.06228)
- **源文件**: `(ECCV 2024) RCM.py`

### 设计机制
- rectangular self-calibration attention (RCA)
- [N, D, C, 1]
- Rectangular Self-Calibration Module (RCM)

## 2. 核心分析
### 类定义与参数
#### `class ConvMlp`
- **描述**: 使用 1x1 卷积保持空间维度的 MLP
    
- **初始化参数**: `in_features, hidden_features, out_features, act_layer, norm_layer, bias, drop`

#### `class RCA`
- **描述**: 无文档说明。
- **初始化参数**: `inp, kernel_size, ratio, band_kernel_size, dw_size, padding, stride, square_kernel_size, relu`

#### `class RCM`
- **描述**: MetaNeXtBlock 块
参数:
    dim (int): 输入通道数.
    drop_path (float): 随机深度率。默认: 0.0
    ls_init_value (float): 层级比例初始化值。默认: 1e-6.
- **初始化参数**: `dim, token_mixer, norm_layer, mlp_layer, mlp_ratio, act_layer, ls_init_value, drop_path, dw_size, square_kernel_size, ratio`

## 3. 使用示例
```python
# 导入方式（参考）：from (ECCV 2024) RCM import ...

input_tensor = torch.randn(1, 64, 32, 32)#输入 B C H W

    # 实例化 RCM 模块
    block = RCM(dim=64)

    # 打印输入的形状
    print(input_tensor.size())

    # 将输入张量传递给 RCM 模块，并打印输出形状
    output_tensor = block(input_tensor)
    print(output_tensor.size())
```

## 4. 适用任务
- **语义分割/实例分割**
- **目标检测**
- **图像分类**
- **图像去噪**
- **目标跟踪**
- **去雨**
- **去雾**
- **去模糊**
- **图像融合**
- **超分辨率**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
