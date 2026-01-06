# Reciprocal Attention Mixing Transformer for Lightweight Image Restoration(CVPR 2024 Workshop)

## 1. 模块简介
- **论文地址**: [https://github.com/rami0205/RAMiT](https://github.com/rami0205/RAMiT)
- **源文件**: `(CVPR 2024) RAMiT.py`

### 设计机制
- RAMiT(Reciprocal Attention Mixing Transformer)
- get pair-wise relative position index for each token inside the window
- define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
- get pair-wise relative position index for each token inside the window

## 2. 核心分析
### 类定义与参数
#### `class DropPath`
- **描述**: Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
- **初始化参数**: `drop_prob`

#### `class QKVProjection`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_head, qkv_bias`

#### `class SpatialSelfAttention`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_head, total_head, window_size, shift, attn_drop, proj_drop, helper`

#### `class ChannelSelfAttention`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_head, total_head, attn_drop, proj_drop, helper`

#### `class ReshapeLayerNorm`
- **描述**: 无文档说明。
- **初始化参数**: `dim, norm_layer`

#### `class MobiVari1`
- **描述**: 无文档说明。
- **初始化参数**: `dim, kernel_size, stride, act, out_dim`

#### `class MobiVari2`
- **描述**: 无文档说明。
- **初始化参数**: `dim, kernel_size, stride, act, out_dim, exp_factor, expand_groups`

#### `class FeedForward`
- **描述**: 无文档说明。
- **初始化参数**: `dim, hidden_ratio, act_layer, bias, drop`

#### `class NoLayer`
- **描述**: 无文档说明。

#### `class DRAMiTransformer`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_head, chsa_head_ratio, window_size, shift, head_dim, qkv_bias, mv_ver, hidden_ratio, act_layer, norm_layer, attn_drop, proj_drop, drop_path, helper, mv_act, exp_factor, expand_groups`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2024) RAMiT import ...

# Instantiate the model
    block = DRAMiTransformer(dim=64)

    input = torch.randn(4, 64, 32, 32) # 输入B C H W

    # Forward pass
    output, sp, ch, attn0 = block(input)

    # Print input and output shapes
    print(input.size())
    print(output.size())
    print(sp.size())
    print(ch.size())
    print(attn0.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
