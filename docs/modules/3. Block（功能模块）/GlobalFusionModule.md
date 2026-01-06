# MAGNet: Multi-scale Awareness and Global fusion Network for RGB-D salient object detection | KBS

## 1. 模块简介
- **论文地址**: [https://www.sciencedirect.com/science/article/abs/pii/S0950705124007603](https://www.sciencedirect.com/science/article/abs/pii/S0950705124007603)
- **源文件**: `GlobalFusionModule.py`

### 设计机制
- Global Fusion Module
- 实例化 GFM 模块
- 前向传播，计算输出
- 打印输入和输出的形状

## 2. 核心分析
### 类定义与参数
#### `class DWPWConv`
- **描述**: 无文档说明。
- **初始化参数**: `inc, outc`

#### `class SAttention`
- **描述**: 无文档说明。
- **初始化参数**: `dim, sa_num_heads, qkv_bias, qk_scale, attn_drop, proj_drop`

#### `class GFM`
- **描述**: 无文档说明。
- **初始化参数**: `inc, expend_ratio`

## 3. 使用示例
```python
# 导入方式（参考）：from GlobalFusionModule import ...

# 实例化 GFM 模块
    inc = 64  # 输入通道数
    block = GFM(inc=inc, expend_ratio=2)

    x = torch.randn(1, inc, 32, 32)  # B  C  H   W
    d = torch.randn(1, inc, 32, 32)  # 与 x 相同形状的深度图

    # 前向传播，计算输出
    output = block(x, d)

    # 打印输入和输出的形状
    print(f"Input x shape: {x.size()}")
    print(f"Input d shape: {d.size()}")
    print(f"Output shape: {output.size()}")
```

## 4. 适用任务
- **目标检测**
- **图像融合**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
