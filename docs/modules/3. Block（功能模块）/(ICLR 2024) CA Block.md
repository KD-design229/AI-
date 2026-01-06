# MogaNet: Multi-order Gated Aggregation Network (ICLR 2024)

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2211.03295](https://arxiv.org/pdf/2211.03295)
- **源文件**: `(ICLR 2024) CA Block.py`

### 设计机制
- FFN with Channel Aggregation
- Build activation layer
- A learnable element-wise scaler.
- x_d: [B, C, H, W] -> [B, 1, H, W]
- proj 1

## 2. 核心分析
### 类定义与参数
#### `class ElementScale`
- **描述**: 无文档说明。
- **初始化参数**: `embed_dims, init_value, requires_grad`

#### `class ChannelAggregationFFN`
- **描述**: An implementation of FFN with Channel Aggregation.

Args:
    embed_dims (int): The feature dimension. Same as
        `MultiheadAttention`.
    feedforward_channels (int): The hidden dimension of FFNs.
    kernel_size (int): The depth-wise conv kernel size as the
        depth-wise convolution. Defaults to 3.
    act_type (str): The type of activation. Defaults to 'GELU'.
    ffn_drop (float, optional): Probability of an element to be
        zeroed in FFN. Default 0.0.
- **初始化参数**: `embed_dims, kernel_size, act_type, ffn_drop`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICLR 2024) CA Block import ...

input = torch.randn(1, 64, 32, 32)# 输入 B C H W
    block = ChannelAggregationFFN(embed_dims=64)
    output = block(input)
    print(input.size())
    print(output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
