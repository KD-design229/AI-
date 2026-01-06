# FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection

## 1. 模块简介
- **论文地址**: [https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848](https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848)
- **源文件**: `FACMA.py`

### 设计机制
- context attention
- 假设的RGB和深度输入

## 2. 核心分析
### 类定义与参数
#### `class FCABlock`
- **描述**: 无文档说明。
- **初始化参数**: `channel, width, height, fidx_u, fidx_v, reduction`

#### `class SFCA`
- **描述**: 无文档说明。
- **初始化参数**: `in_channel, width, height, fidx_u, fidx_v`

#### `class FACMA`
- **描述**: 无文档说明。
- **初始化参数**: `in_channel, width, height, fidx_u, fidx_v`

## 3. 使用示例
```python
# 导入方式（参考）：from FACMA import ...

# 定义输入参数
    in_channel = 64
    width = 224
    height = 224
    fidx_u = [0, 1]
    fidx_v = [0, 1]

    block = FACMA(in_channel, width, height, fidx_u, fidx_v)

    # 假设的RGB和深度输入
    rgb_input = torch.randn(1, in_channel, width, height)  # Batch size为1
    depth_input = torch.randn(1, in_channel, width, height)  # Batch size为1

    # 通过FACMA
    out_rgb, out_d = block(rgb_input, depth_input)

    # 打印输入输出形状
    print("RGB 输入形状:", rgb_input.shape)
    print("深度 输入形状:", depth_input.shape)
    print("RGB 输出形状:", out_rgb.shape)
    print("深度 输出形状:", out_d.shape)
```

## 4. 适用任务
- **目标检测**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
