# (ACCV 2024) WMB

## 1. 模块简介
- **源文件**: `(ACCV 2024) WMB.py`

### 设计机制
- 缩小hw和扩大b到4b
- 使用哈尔 haar 小波变换来实现二维离散小波
- 还原 b和hw

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

#### `class DWT`
- **描述**: 无文档说明。

#### `class IWT`
- **描述**: 无文档说明。

#### `class Conv2d_BN`
- **描述**: 无文档说明。
- **初始化参数**: `a, b, ks, stride, pad, dilation, groups, bn_weight_init, resolution`

#### `class FeedForward`
- **描述**: 无文档说明。
- **初始化参数**: `dim, ffn_expansion_factor, bias`

#### `class WM`
- **描述**: 无文档说明。
- **初始化参数**: `c`

#### `class Illumination_Estimator`
- **描述**: 无文档说明。
- **初始化参数**: `n_fea_middle, n_fea_in, n_fea_out`

#### `class WMB`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type`

## 3. 使用示例
```python
# 导入方式（参考）：from (ACCV 2024) WMB import ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1  # Batch size
    channels = 32   # 输入通道数
    height = 256    # 输入图像高度
    width = 256     # 输入图像宽度

    # 创建一个模拟输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width).to(device)

    # 初始化 WMB 模块
    model = WMB(dim=channels, num_heads=8, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias').to(device)
    print(model)
    print("微信公众号: AI缝合术!")
    # 前向传播
    output = model(x)

    # 打印输入和输出张量的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
