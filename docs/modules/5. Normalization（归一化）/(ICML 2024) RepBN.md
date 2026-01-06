# SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2405.11582](https://arxiv.org/pdf/2405.11582)
- **源文件**: `(ICML 2024) RepBN.py`

### 设计机制
- 中文题目：SLAB：具有简化线性注意力和渐进式重参数化批量归一化的高效变换器
- 官方github：https://github.com/xinghaochen/SLAB
- 所属机构：华为诺亚方舟实验室
- 源代码, 处理三维数据
- BatchNorm2d处理四维输入数据
- if __name__ == "__main__":
- # 模块参数
- batch_size = 1    # 批大小
- channels = 32     # 输入特征通道数
- N = 16 * 16      # 图像高度*宽度 height * width
- model = RepBN(channels = channels)
- print(model)
- # 生成随机输入张量 (batch_size, channels, height * width (N))
- x = torch.randn(batch_size, N, channels)
- # 打印输入张量的形状

## 2. 核心分析
### 类定义与参数
#### `class RepBN`
- **描述**: 无文档说明。
- **初始化参数**: `channels`

#### `class RepBN2d`
- **描述**: 无文档说明。
- **初始化参数**: `channels`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICML 2024) RepBN import ...

#     # 模块参数
#     batch_size = 1    # 批大小
#     channels = 32     # 输入特征通道数
#     N = 16 * 16      # 图像高度*宽度 height * width

#     model = RepBN(channels = channels)
#     print(model)
#     print("微信公众号:AI缝合术, nb!")

#     # 生成随机输入张量 (batch_size, channels, height * width (N))
#     x = torch.randn(batch_size, N, channels)
#     # 打印输入张量的形状
#     print("Input shape:", x.shape)
#     # 前向传播计算输出
#     output = model(x)
#     # 打印输出张量的形状
#     print("Output shape:", output.shape)

if __name__ == "__main__":
    # 模块参数
    batch_size = 1    # 批大小
    channels = 32     # 输入特征通道数
    height = 256      # 图像高度
    width = 256        # 图像宽度

    model = RepBN2d(channels = channels)
    print(model)
    print("微信公众号:AI缝合术, nb!")

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    # 前向传播计算输出
    output = model(x)
    # 打印输出张量的形状
    print("Output shape:", output.shape)
```

## 4. 适用任务
- **Transformer相关任务**
- **注意力机制应用**
- **归一化/模型稳定化**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
