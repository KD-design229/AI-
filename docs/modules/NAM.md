# NAM: Normalization-based Attention Module

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2111.12419](https://arxiv.org/pdf/2111.12419)
- **源文件**: `NAM.py`

### 设计机制
- from torchinfo import summary  # 计算参数量，可注释
- from fvcore.nn import FlopCountAnalysis, flop_count_table  # 计算计算量，可注释
- 中文题目:  NAM： 基于归一化的注意力模块
- 官方github：https://github.com/Christian-lyc/NAM
- 所属机构：东北大学医学院和生物信息工程学院等
- 关键词：归一化注意力、空间注意力、通道注意力、图像分类
- 输入特征：x，形状为 (B, C, H, W)
- 计算每个像素的权重 (Pixel Normalization)
- # 参数量计算，可注释
- print("Model Summary:")
- print(summary(nam, input_size=(1, 32, 256, 256), device=device.type))
- # 计算量计算，可注释
- flops = FlopCountAnalysis(nam, input)
- print("\nFlop Count Table:")
- print(flop_count_table(flops))

## 2. 核心分析
### 类定义与参数
#### `class Channel_Att`
- **描述**: 无文档说明。
- **初始化参数**: `channels, t`

#### `class Spatial_Att`
- **描述**: 无文档说明。
- **初始化参数**: `kernel_size`

#### `class NAM`
- **描述**: 无文档说明。
- **初始化参数**: `channels`

## 3. 使用示例
```python
# 导入方式（参考）：from NAM import ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nam = NAM(channels=32).to(device)  # 将模型移动到设备
    input = torch.rand(1, 32, 256, 256).to(device)  # 将输入数据移动到设备

    # # 参数量计算，可注释
    # print("Model Summary:")
    # print(summary(nam, input_size=(1, 32, 256, 256), device=device.type))

    # # 计算量计算，可注释
    # flops = FlopCountAnalysis(nam, input)
    # print("\nFlop Count Table:")
    # print(flop_count_table(flops))

    output = nam(input)
    print(f"\nInput shape: {input.shape}")
    print(f"Output shape: {output.shape}")
```

## 4. 适用任务
- **图像分类**
- **医学图像处理**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
