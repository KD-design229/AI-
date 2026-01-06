# Aggregating Global Features into Local Vision Transformer

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2201.12903](https://arxiv.org/pdf/2201.12903)
- **源文件**: `(ICPR 2022) MOATransformer.py`

### 设计机制
- define a parameter table of relative position bias

## 2. 核心分析
### 类定义与参数
#### `class Mlp`
- **描述**: 无文档说明。
- **初始化参数**: `in_features, hidden_features, out_features, act_layer, drop`

#### `class WindowAttention`
- **描述**: Window based multi-head self attention (W-MSA) module with relative position bias.
It supports both of shifted and non-shifted window.

Args:
    dim (int): Number of input channels.
    window_size (tuple[int]): The height and width of the window.
    num_heads (int): Number of attention heads.
    qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    proj_drop (float, optional): Dropout ratio of output. Default: 0.0
- **初始化参数**: `dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop`

#### `class GlobalAttention`
- **描述**: MOA - multi-head self attention (W-MSA) module with relative position bias.

Args:
    dim (int): Number of input channels.
    window_size (tuple[int]): The height and width of the window.
    num_heads (int): Number of attention heads.
    qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    proj_drop (float, optional): Dropout ratio of output. Default: 0.0
- **初始化参数**: `dim, window_size, input_resolution, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop`

#### `class LocalTransformerBlock`
- **描述**: Local Transformer Block.

Args:
    dim (int): Number of input channels.
    input_resolution (tuple[int]): Input resulotion.
    num_heads (int): Number of attention heads.
    window_size (int): Window size.
    shift_size (int): Shift size for SW-MSA.
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
    drop (float, optional): Dropout rate. Default: 0.0
    attn_drop (float, optional): Attention dropout rate. Default: 0.0
    drop_path (float, optional): Stochastic depth rate. Default: 0.0
    act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
    norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
- **初始化参数**: `dim, input_resolution, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer`

#### `class PatchMerging`
- **描述**: Patch Merging Layer.

Args:
    input_resolution (tuple[int]): Resolution of input feature.
    dim (int): Number of input channels.
    norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
- **初始化参数**: `input_resolution, dim, norm_layer`

#### `class BasicLayer`
- **描述**: A basic Swin Transformer layer for one stage.

Args:
    dim (int): Number of input channels.
    input_resolution (tuple[int]): Input resolution.
    depth (int): Number of blocks.
    num_heads (int): Number of attention heads.
    window_size (int): Local window size.
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
    drop (float, optional): Dropout rate. Default: 0.0
    attn_drop (float, optional): Attention dropout rate. Default: 0.0
    drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
    norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
- **初始化参数**: `dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, downsample, drop_path_global, use_checkpoint`

#### `class PatchEmbed`
- **描述**: Image to Patch Embedding

Args:
    img_size (int): Image size.  Default: 224.
    patch_size (int): Patch token size. Default: 4.
    in_chans (int): Number of input image channels. Default: 3.
    embed_dim (int): Number of linear projection output channels. Default: 96.
    norm_layer (nn.Module, optional): Normalization layer. Default: None
- **初始化参数**: `img_size, patch_size, in_chans, embed_dim, norm_layer`

#### `class MOATransformer`
- **描述**: Swin Transformer
    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
      https://arxiv.org/pdf/2103.14030

Args:
    img_size (int | tuple(int)): Input image size. Default 224
    patch_size (int | tuple(int)): Patch size. Default: 4
    in_chans (int): Number of input image channels. Default: 3
    num_classes (int): Number of classes for classification head. Default: 1000
    embed_dim (int): Patch embedding dimension. Default: 96
    depths (tuple(int)): Depth of each Swin Transformer layer.
    num_heads (tuple(int)): Number of attention heads in different layers.
    window_size (int): Window size. Default: 7
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
    qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
    drop_rate (float): Dropout rate. Default: 0
    attn_drop_rate (float): Attention dropout rate. Default: 0
    drop_path_rate (float): Stochastic depth rate. Default: 0.1
    norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
    patch_norm (bool): If True, add normalization after patch embedding. Default: True
    use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
- **初始化参数**: `img_size, patch_size, in_chans, num_classes, embed_dim, depths, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICPR 2022) MOATransformer import ...

input=torch.randn(1,3,224,224)
    model = MOATransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6],
        num_heads=[3, 6, 12],
        window_size=14,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )
    output=model(input)
    print(output.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
