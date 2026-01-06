# Demystify Mamba in Vision: A Linear Attention Perspective

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2405.16605](https://arxiv.org/pdf/2405.16605)
- **源文件**: `(NeurIPS 2024) MLLA.py`

### 设计机制
- 中文题目：在视觉中揭开曼巴的神秘面纱：一种线性注意力视角
- 官方github：https://github.com/LeapLabTHU/MLLA
- 所属机构：清华大学，阿里巴巴集团
- q, k, v: b, n, c

## 2. 核心分析
### 类定义与参数
#### `class Mlp`
- **描述**: 无文档说明。
- **初始化参数**: `in_features, hidden_features, out_features, act_layer, drop`

#### `class RoPE`
- **描述**: Rotary Positional Embedding.
    
- **初始化参数**: `shape, base`

#### `class LinearAttention`
- **描述**: Linear Attention with LePE and RoPE.

Args:
    dim (int): Number of input channels.
    num_heads (int): Number of attention heads.
    qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
- **初始化参数**: `dim, input_resolution, num_heads, qkv_bias`

#### `class MLLABlock`
- **描述**: MLLA Block.
Args:
    dim (int): Number of input channels.
    input_resolution (tuple[int]): Input resulotion.
    num_heads (int): Number of attention heads.
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    drop (float, optional): Dropout rate. Default: 0.0
    drop_path (float, optional): Stochastic depth rate. Default: 0.0
    act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
    norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
- **初始化参数**: `dim, input_resolution, num_heads, mlp_ratio, qkv_bias, drop, drop_path, act_layer, norm_layer`

## 3. 使用示例
```python
# 导入方式（参考）：from (NeurIPS 2024) MLLA import ...

# 模块参数
    batch_size = 1     # 批大小
    channels = 96      # 输入特征通道数
    height = 224        # 图像高度
    width = 224         # 图像宽度
    N =height * width  # 序列长度

    model = MLLABlock(dim=channels, input_resolution=(height,width), num_heads=3)
    print(model)
    print("微信公众号:AI缝合术")

    # 生成随机输入张量 (batch_size,height*width, channels)
    x = torch.randn(batch_size, N, channels)
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    # 前向传播计算输出
    output = model(x)
    # 打印输出张量的形状
    print("Output shape:", output.shape)
```

## 4. 适用任务
- **注意力机制应用**
- **长序列建模/Mamba应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
