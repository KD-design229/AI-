# CoAtNet: Marrying Convolution and Attention for All Data Sizes

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2106.04803](https://arxiv.org/pdf/2106.04803)
- **源文件**: `(NeurIPS 2021) CoAtNet.py`

## 2. 核心分析
### 类定义与参数
#### `class ScaledDotProductAttention`
- **描述**: Scaled dot-product attention
- **初始化参数**: `d_model, d_k, d_v, h, dropout`

#### `class SwishImplementation`
- **描述**: 无文档说明。

#### `class MemoryEfficientSwish`
- **描述**: 无文档说明。

#### `class Conv2dStaticSamePadding`
- **描述**: 2D Convolutions like TensorFlow, for a fixed image size
- **初始化参数**: `in_channels, out_channels, kernel_size, image_size`

#### `class Identity`
- **描述**: 无文档说明。

#### `class MBConvBlock`
- **描述**: 层 ksize3*3 输入32 输出16  conv1  stride步长1
- **初始化参数**: `ksize, input_filters, output_filters, expand_ratio, stride, image_size`

#### `class CoAtNet`
- **描述**: 无文档说明。
- **初始化参数**: `in_ch, image_size, out_chs`

## 3. 使用示例
```python
# 导入方式（参考）：from (NeurIPS 2021) CoAtNet import ...

x=torch.randn(1,3,224,224)
    coatnet=CoAtNet(3,224)
    y=coatnet(x)
    print(y.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
