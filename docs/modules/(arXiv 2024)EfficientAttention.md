# Efficient Attention: Attention with Linear Complexities

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/1812.01243](https://arxiv.org/pdf/1812.01243)
- **源文件**: `(arXiv 2024)EfficientAttention.py`

### 设计机制
- 中文题目:  高效注意力：具有线性复杂度的注意力机制
- 官方github：https://github.com/cmsflash/efficient-attention
- 代码整理与注释：公众号：AI缝合术
- AI缝合术github：https://github.com/AIFengheshu/Plug-play-modules
- EfficientAttention 模块：一个高效的多头注意力机制
- 1x1 卷积层，用于生成键（keys）、查询（queries）和值（values）
- 最后的 1x1 卷积用于将注意力输出映射回输入通道数
- 计算每个头的键通道数和值通道数
- 对每个头进行注意力计算
- 从键和值中提取当前头的部分
- 最终输出是加上输入的残差连接
- 测试 EfficientAttention 模块
- 创建一个 EfficientAttention 实例
- 创建一个随机输入张量，模拟输入特征图 (batch_size=1, in_channels=64, height=32, width=32)
- 通过注意力模块进行前向传播

## 2. 核心分析
### 类定义与参数
#### `class EfficientAttention`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, key_channels, head_count, value_channels`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2024)EfficientAttention import ...

# 创建一个 EfficientAttention 实例
    attention = EfficientAttention(in_channels=64, key_channels=128, head_count=4, value_channels=128)
    
    # 创建一个随机输入张量，模拟输入特征图 (batch_size=1, in_channels=64, height=32, width=32)
    input_tensor = torch.randn(1, 64, 32, 32)

    # 通过注意力模块进行前向传播
    output = attention(input_tensor)
    
    # 打印输入和输出张量的形状
    print(f'输入形状: {input_tensor.shape}')
    print(f'输出形状: {output.shape}')
```

## 4. 适用任务
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
