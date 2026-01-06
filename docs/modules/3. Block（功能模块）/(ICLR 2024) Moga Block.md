# MogaNet: Multi-order Gated Aggregation Network (ICLR 2024)

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2211.03295](https://arxiv.org/pdf/2211.03295)
- **源文件**: `(ICLR 2024) Moga Block.py`

### 设计机制
- Spatial Block with Multi-order Gated Aggregation
- basic DW conv
- DW conv 1
- DW conv 2
- a channel convolution

## 2. 核心分析
### 类定义与参数
#### `class ElementScale`
- **描述**: 无文档说明。
- **初始化参数**: `embed_dims, init_value, requires_grad`

#### `class MultiOrderDWConv`
- **描述**: 无文档说明。
- **初始化参数**: `embed_dims, dw_dilation, channel_split`

#### `class MultiOrderGatedAggregation`
- **描述**: 无文档说明。
- **初始化参数**: `embed_dims, attn_dw_dilation, attn_channel_split, attn_act_type, attn_force_fp32`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICLR 2024) Moga Block import ...

input = torch.randn(1, 64, 32, 32)# 输入 B C H W
    block = MultiOrderGatedAggregation(embed_dims=64)
    output = block(input)
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
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
