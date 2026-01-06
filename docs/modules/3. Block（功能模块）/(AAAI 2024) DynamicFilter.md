# FFT-based Dynamic Token Mixer for Vision

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2303.03932](https://arxiv.org/pdf/2303.03932)
- **源文件**: `(AAAI 2024) DynamicFilter.py`

## 2. 核心分析
### 类定义与参数
#### `class StarReLU`
- **描述**: StarReLU: s * relu(x) ** 2 + b
- **初始化参数**: `scale_value, bias_value, scale_learnable, bias_learnable, mode, inplace`

#### `class Mlp`
- **描述**: MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
Mostly copied from timm.
- **初始化参数**: `dim, mlp_ratio, out_features, act_layer, drop, bias`

#### `class DynamicFilter`
- **描述**: 无文档说明。
- **初始化参数**: `dim, expansion_ratio, reweight_expansion_ratio, act1_layer, act2_layer, bias, num_filters, size, weight_resize`

## 3. 使用示例
```python
# 导入方式（参考）：from (AAAI 2024) DynamicFilter import ...

block = DynamicFilter(32, size=64)  # size==H,W
    print(block)
    print("微信公众号: AI缝合术!")

    # 若input形状为B C H W，先用下面代码变换张量形状
    input = torch.rand(3, 32, 64, 64)   # 输入 B C H W
    input_bhwc = input.permute(0, 2, 3, 1)  # B H W C

    output = block(input_bhwc)

    output = output.permute(0, 3, 1, 2)  # B C H W
    print(input.size())
    print(output.size())  # 输出的形状
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
- **Transformer相关任务**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
