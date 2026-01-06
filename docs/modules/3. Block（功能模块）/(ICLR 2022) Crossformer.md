# CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention (ICLR 2022 Acceptance).

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2108.00154](https://arxiv.org/pdf/2108.00154)
- **源文件**: `(ICLR 2022) Crossformer.py`

## 2. 核心分析
### 类定义与参数
#### `class Mlp`
- **描述**: 无文档说明。
- **初始化参数**: `in_features, hidden_features, out_features, act_layer, drop`

#### `class DynamicPosBias`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_heads, residual`

#### `class Attention`
- **描述**: Multi-head self attention module with dynamic position bias.

Args:
    dim (int): Number of input channels.
    group_size (tuple[int]): The height and width of the group.
    num_heads (int): Number of attention heads.
    qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    proj_drop (float, optional): Dropout ratio of output. Default: 0.0
- **初始化参数**: `dim, group_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, position_bias`

#### `class CrossFormerBlock`
- **描述**: CrossFormer Block.

Args:
    dim (int): Number of input channels.
    input_resolution (tuple[int]): Input resulotion.
    num_heads (int): Number of attention heads.
    group_size (int): Group size.
    lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
    drop (float, optional): Dropout rate. Default: 0.0
    attn_drop (float, optional): Attention dropout rate. Default: 0.0
    drop_path (float, optional): Stochastic depth rate. Default: 0.0
    act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
    norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
- **初始化参数**: `dim, input_resolution, num_heads, group_size, lsda_flag, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer, num_patch_size`

#### `class PatchMerging`
- **描述**: Patch Merging Layer.

Args:
    input_resolution (tuple[int]): Resolution of input feature.
    dim (int): Number of input channels.
    norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
- **初始化参数**: `input_resolution, dim, norm_layer, patch_size, num_input_patch_size`

#### `class Stage`
- **描述**: CrossFormer blocks for one stage.

Args:
    dim (int): Number of input channels.
    input_resolution (tuple[int]): Input resolution.
    depth (int): Number of blocks.
    num_heads (int): Number of attention heads.
    group_size (int): variable G in the paper, one group has GxG embeddings
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
    drop (float, optional): Dropout rate. Default: 0.0
    attn_drop (float, optional): Attention dropout rate. Default: 0.0
    drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
    norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
- **初始化参数**: `dim, input_resolution, depth, num_heads, group_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer, downsample, use_checkpoint, patch_size_end, num_patch_size`

#### `class PatchEmbed`
- **描述**: Image to Patch Embedding

Args:
    img_size (int): Image size.  Default: 224.
    patch_size (int): Patch token size. Default: [4].
    in_chans (int): Number of input image channels. Default: 3.
    embed_dim (int): Number of linear projection output channels. Default: 96.
    norm_layer (nn.Module, optional): Normalization layer. Default: None
- **初始化参数**: `img_size, patch_size, in_chans, embed_dim, norm_layer`

#### `class CrossFormer`
- **描述**: CrossFormer
    A PyTorch impl of : `CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention`  -

Args:
    img_size (int | tuple(int)): Input image size. Default 224
    patch_size (int | tuple(int)): Patch size. Default: 4
    in_chans (int): Number of input image channels. Default: 3
    num_classes (int): Number of classes for classification head. Default: 1000
    embed_dim (int): Patch embedding dimension. Default: 96
    depths (tuple(int)): Depth of each stage.
    num_heads (tuple(int)): Number of attention heads in different layers.
    group_size (int): Group size. Default: 7
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
- **初始化参数**: `img_size, patch_size, in_chans, num_classes, embed_dim, depths, num_heads, group_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint, merge_size`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICLR 2022) Crossformer import ...

input=torch.randn(1,3,224,224)
    model = CrossFormer(img_size=224,
        patch_size=[4, 8, 16, 32],
        in_chans= 3,
        num_classes=1000,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_size=[7, 7, 7, 7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2,4], [2, 4]]
    )
    output=model(input)
    print(output.shape)
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
