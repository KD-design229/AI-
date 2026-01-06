# ABC: Attention with Bilinear Correlation for Infrared Small Target Detection ICME2023

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2303.10321](https://arxiv.org/pdf/2303.10321)
- **源文件**: `(ICME 2023) CLFT.py`

### 设计机制
- bilinear attention module (BAM)
- dilated convolution layers(DConv)
- self.x_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
- convolution linear fusion transformer (CLFT)

## 2. 核心分析
### 类定义与参数
#### `class BAM`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim, in_feature, out_feature`

#### `class Conv`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim`

#### `class DConv`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim`

#### `class ConvAttention`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim, in_feature, out_feature`

#### `class FeedForward`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim, out_dim`

#### `class CLFT`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim, out_dim, in_feature, out_feature`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICME 2023) CLFT import ...

block = CLFT(64,64,32*32,32) # 输入通道数，输出通道数 图像大小 H*W，H or W


    input = torch.randn(3, 64, 32, 32)   #输入tensor形状 B C H W

    # Print input shape
    print(input.size()) # 输入形状

    # Pass the input tensor through the model
    output = block(input)

    # Print output shape
    print(output.size()) # 输出形状
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
