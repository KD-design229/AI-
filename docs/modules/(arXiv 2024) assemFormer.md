# SvANet: A Scale-variant Attention-based Network for Small Medical Object Segmentation

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2407.07720v1](https://arxiv.org/pdf/2407.07720v1)
- **源文件**: `(arXiv 2024) assemFormer.py`

### 设计机制
- Make sure that round down does not go down by more than 10%.
- [B, C, P, N] --> [B, h + 2d, P, N]
- Project x into query, key and value
- Query --> [B, 1, P, N]
- value, key --> [B, d, P, N]
- apply softmax along N dimension
- Uncomment below line to visualize context scores
- self.visualize_context_scores(context_scores=context_scores)
- Compute context vector
- [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]

## 2. 核心分析
### 类定义与参数
#### `class Dropout`
- **描述**: 无文档说明。
- **初始化参数**: `p, inplace`

#### `class StochasticDepth`
- **描述**: 无文档说明。
- **初始化参数**: `p, Mode`

#### `class LinearSelfAttention`
- **描述**: This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
This layer can be used for self- as well as cross-attention.

Args:
    opts: command line arguments
    DimEmbed (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
    AttnDropRate (Optional[float]): Dropout value for context scores. Default: 0.0
    bias (Optional[bool]): Use bias in learnable layers. Default: True

Shape:
    - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
    :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
    - Output: same as the input

.. note::
    For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
    in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
    we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
    expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
    channel-first to channel-last format in case of a linear layer.
- **初始化参数**: `DimEmbed, AttnDropRate, Bias`

#### `class LinearAttnFFN`
- **描述**: 无文档说明。
- **初始化参数**: `DimEmbed, DimFfnLatent, AttnDropRate, DropRate, FfnDropRate`

#### `class BaseConv2d`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, kernel_size, stride, padding, groups, bias, BNorm, ActLayer, dilation, Momentum`

#### `class BaseFormer`
- **描述**: 无文档说明。
- **初始化参数**: `InChannels, FfnMultiplier, NumAttnBlocks, AttnDropRate, DropRate, FfnDropRate, PatchRes, Dilation, ViTSELayer`

#### `class AssemFormer`
- **描述**: Inspired by MobileViTv3.
Adapted from https://github.com/micronDLA/MobileViTv3/blob/main/MobileViTv3-v2/cvnets/modules/mobilevit_block.py
- **初始化参数**: `InChannels, FfnMultiplier, NumAttnBlocks, AttnDropRate, DropRate, FfnDropRate, PatchRes, Dilation, SDProb, ViTSELayer`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2024) assemFormer import ...

input = torch.randn(1, 64, 128, 128)# 输入 B C H W
    block = AssemFormer(InChannels=64)
    output = block(input)
    print(output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
