# Restoring Images in Adverse Weather Conditions via Histogram Transformer (ECCV 2024)

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2407.10172](https://arxiv.org/pdf/2407.10172)
- **源文件**: `(ECCV 2024) HTB.py`

### 设计机制
- Layer Norm
- return rearrange(x, 'b c h w -> b c (h w)')
- return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
- Dual-scale Gated Feed-Forward Network (DGFF)

## 2. 核心分析
### 类定义与参数
#### `class BiasFree_LayerNorm`
- **描述**: 无文档说明。
- **初始化参数**: `normalized_shape`

#### `class WithBias_LayerNorm`
- **描述**: 无文档说明。
- **初始化参数**: `normalized_shape`

#### `class LayerNorm`
- **描述**: 无文档说明。
- **初始化参数**: `dim, LayerNorm_type`

#### `class FeedForward`
- **描述**: 无文档说明。
- **初始化参数**: `dim, ffn_expansion_factor, bias`

#### `class Attention_histogram`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_heads, bias, ifBox`

#### `class TransformerBlock`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type`

## 3. 使用示例
```python
# 导入方式（参考）：from (ECCV 2024) HTB import ...

input = torch.randn(1, 64, 128, 128)#输入 B C H W


    transformer_block = TransformerBlock(64)#输入C

    # 前向传播
    output = transformer_block(input)

    # 打印输入和输出的形状
    print(input.size())
    print(output.size())
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
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
