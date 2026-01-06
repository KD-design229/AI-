# Fully Attentional Network for Semantic Segmentation

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2112.04108](https://arxiv.org/pdf/2112.04108)
- **源文件**: `FullyAttentional.py`

### 设计机制
- 官方github：https://github.com/maggiesong7/FullyAttentional?tab=readme-ov-file
- 代码整理与注释：公众号：AI缝合术
- AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules
- https://github.com/Ilareina/FullyAttentional/blob/main/model.py
- 初始化函数，plane是输入和输出特征图的通道数，norm_layer是归一化层（默认为BatchNorm2d）
- 定义两个全连接层，conv1和conv2
- 定义卷积层 + 归一化层 + 激活函数（ReLU）
- 定义softmax操作，用于计算关系矩阵
- 初始化可学习的参数gamma，用于调整最终的输出
- 前向传播过程，x为输入的特征图，形状为 (batch_size, channels, height, width)
- 对输入张量进行排列和变形，获取水平和垂直方向的特征
- 对输入张量分别在水平方向和垂直方向进行池化，并通过全连接层进行编码
- 计算水平方向和垂直方向的关系矩阵
- 计算经过softmax后的关系矩阵
- 通过矩阵乘法和关系矩阵，对特征进行加权和增强

## 2. 核心分析
### 类定义与参数
#### `class FullyAttentionalBlock`
- **描述**: 无文档说明。
- **初始化参数**: `plane, norm_layer`

## 3. 使用示例
```python
# 导入方式（参考）：from FullyAttentional import ...

fab = FullyAttentionalBlock(plane=32).cuda()
    # 随机生成输入张量 (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256).cuda()
    # 打印输入张量的形状
    print(f"输入张量的形状: {input_tensor.shape}")
    # 前向传播
    output_tensor = fab(input_tensor)
    # 打印输出张量的形状
    print(f"输出张量的形状: {output_tensor.shape}")
```

## 4. 适用任务
- **语义分割/实例分割**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
