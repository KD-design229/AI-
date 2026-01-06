# BAM: Bottleneck Attention Module

## 1. 模块简介
- **论文地址**: [http://bmvc2018.org/contents/papers/0092.pdf](http://bmvc2018.org/contents/papers/0092.pdf)
- **源文件**: `BAM.py`

### 设计机制
- 中文题目:  BAM：瓶颈注意力模块
- 官方github：https://github.com/Jongchan/attention-module
- 所属机构：Lunit Inc., 韩国; 韩国科学技术院(KAIST), 韩国; Adobe Research, 美国
- 关键词： 瓶颈注意力模块, 深度神经网络, 注意力机制, 图像分类, 目标检测
- 随机生成输入张量 (B, C, H, W)
- 打印输入张量的形状
- 打印输出张量的形状

## 2. 核心分析
### 类定义与参数
#### `class Flatten`
- **描述**: 无文档说明。

#### `class ChannelAttention`
- **描述**: 无文档说明。
- **初始化参数**: `channel, reduction, num_layers`

#### `class SpatialAttention`
- **描述**: 无文档说明。
- **初始化参数**: `channel, reduction, num_layers, dia_val`

#### `class BAMBlock`
- **描述**: 无文档说明。
- **初始化参数**: `channel, reduction, dia_val`

## 3. 使用示例
```python
# 导入方式（参考）：from BAM import ...

bam = BAMBlock(channel=32)
    # 随机生成输入张量 (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256)
    # 打印输入张量的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    # 前向传播
    output_tensor = bam(input_tensor)
    # 打印输出张量的形状
    print(f"输出张量的形状: {output_tensor.shape}")
```

## 4. 适用任务
- **目标检测**
- **图像分类**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
