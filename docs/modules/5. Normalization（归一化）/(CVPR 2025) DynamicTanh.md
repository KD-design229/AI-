# (CVPR 2025) DynamicTanh

## 1. 模块简介
- **源文件**: `(CVPR 2025) DynamicTanh.py`

### 设计机制
- 定义需要学习的参数
- 根据 channels_last 参数确定加权方式
- 参数设置
- 创建随机输入张量, 形状为 (batch_size, in_channels, height, width)
- 打印模型结构
- 进行前向传播, 得到输出
- 打印输入和输出的形状

## 2. 核心分析
### 类定义与参数
#### `class DynamicTanh`
- **描述**: 无文档说明。
- **初始化参数**: `normalized_shape, alpha_init_value, channels_last`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2025) DynamicTanh import ...

# 参数设置
    batch_size = 1               # 批量大小
    in_channels = 32             # 输入通道数
    height, width = 256, 256     # 输入图像的高度和宽度

    # 创建随机输入张量, 形状为 (batch_size, in_channels, height, width)
    x = torch.randn(batch_size, in_channels, height, width)

    model = DynamicTanh(normalized_shape=(in_channels, height, width))

    # 打印模型结构
    print(model)
    print("微信公众号: AI缝合术!")

    # 进行前向传播, 得到输出
    output = model(x)

    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
