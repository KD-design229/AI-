# (CVPR 2025) GatedCNNBlock

## 1. 模块简介
- **源文件**: `(CVPR 2025) GatedCNNBlock.py`

### 设计机制
- 创建一个模拟输入张量，形状为 (batch_size, height, width, channels)
- 初始化 GatedCNNBlock 模块
- 打印输入和输出张量的形状

## 2. 核心分析
### 类定义与参数
#### `class GatedCNNBlock`
- **描述**: Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
Args: 
    conv_ratio: control the number of channels to conduct depthwise convolution.
        Conduct convolution on partial channels can improve practical efficiency.
        The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
        also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
- **初始化参数**: `dim, expansion_ratio, kernel_size, conv_ratio, norm_layer, act_layer, drop_path`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2025) GatedCNNBlock import ...

batch_size = 1  # Batch size
    channels = 32   # 输入通道数
    height = 256    # 输入图像高度
    width = 256     # 输入图像宽度

    # 创建一个模拟输入张量，形状为 (batch_size, height, width, channels)
    x = torch.randn(batch_size, height, width, channels)

    # 初始化 GatedCNNBlock 模块
    model = GatedCNNBlock(dim=channels, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0, drop_path=0.1)
    print(model)
    print("微信公众号: AI缝合术!")
    # 前向传播
    output = model(x)

    # 打印输入和输出张量的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

## 4. 适用任务
- **通用视觉任务**: 图像分类、目标检测、语义分割等。
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
