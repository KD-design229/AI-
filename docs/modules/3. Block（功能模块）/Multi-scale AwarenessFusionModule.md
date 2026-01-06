# MAGNet: Multi-scale Awareness and Global fusion Network for RGB-D salient object detection | KBS

## 1. 模块简介
- **论文地址**: [https://www.sciencedirect.com/science/article/abs/pii/S0950705124007603](https://www.sciencedirect.com/science/article/abs/pii/S0950705124007603)
- **源文件**: `Multi-scale AwarenessFusionModule.py`

### 设计机制
- Multi-scale Awareness Fusion Module
- multi = x * d
- B, C, H, W = x.shape
- x_cat = torch.cat((x, d, multi), dim=1)

## 2. 核心分析
### 类定义与参数
#### `class COI`
- **描述**: 无文档说明。
- **初始化参数**: `inc, k, p`

#### `class MHMC`
- **描述**: 无文档说明。
- **初始化参数**: `dim, ca_num_heads, qkv_bias, proj_drop, ca_attention, expand_ratio`

#### `class MAFM`
- **描述**: 无文档说明。
- **初始化参数**: `inc`

## 3. 使用示例
```python
# 导入方式（参考）：from Multi-scale AwarenessFusionModule import ...

inc = 64  # 输入通道数
    block = MAFM(inc=inc)

    # 创建示例输入数据
    x = torch.randn(1, inc, 32, 32)  # B   C   H   W
    d = torch.randn(1, inc, 32, 32)  # 与 x 相同形状的深度图

    # 前向传播，计算输出
    output = block(x, d)

    # 打印输入和输出的形状
    print(f"Input x shape: {x.size()}")
    print(f"Input d shape: {d.size()}")
    print(f"Output shape: {output.size()}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
