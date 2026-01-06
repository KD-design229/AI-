# DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention

## 1. 模块简介
- **论文地址**: [https://github.com/cecret3350/DEA-Net/tree/main](https://github.com/cecret3350/DEA-Net/tree/main)
- **源文件**: `(TIP2024) CGAFusion.py`

### 设计机制
- --------------------------------------------------------
- --------------------------------------------------------

## 2. 核心分析
### 类定义与参数
#### `class SpatialAttention`
- **描述**: 无文档说明。

#### `class ChannelAttention`
- **描述**: 无文档说明。
- **初始化参数**: `dim, reduction`

#### `class PixelAttention`
- **描述**: 无文档说明。
- **初始化参数**: `dim`

#### `class CGAFusion`
- **描述**: 无文档说明。
- **初始化参数**: `dim, reduction`

## 3. 使用示例
```python
# 导入方式（参考）：from (TIP2024) CGAFusion import ...

block = CGAFusion(32)
    input1 = torch.rand(3, 32, 64, 64) # 输入 N C H W
    input2 = torch.rand(3, 32, 64, 64)
    output = block(input1, input2)
    print(output.size())
```

## 4. 适用任务
- **去雾**
- **图像融合**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
