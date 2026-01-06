# Medical Transformer: Gated Axial-Attention forMedical Image Segmentation

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2102.10662](https://arxiv.org/pdf/2102.10662)
- **源文件**: `AxialAttention.py`

### 设计机制
- 中文题目:  医疗Transformer：用于医学图像分割的门控轴向注意力机制
- 官方github：https://github.com/jeya-maria-jose/Medical-Transformer
- 所属机构：约翰霍普金斯大学, 新泽西州立大学
- 关键词： Transformer, 医学图像分割, 自注意力机制
- 定义1x1卷积，用于改变通道数
- 定义一个一维卷积层，用于进行qkv变换
- 定义轴向注意力模块（Axial Attention）
- 确保输入通道数和输出通道数都能被组数整除
- 多头自注意力的qkv变换层
- 如果步长大于1，则添加池化层
- 根据宽度或高度调整输入张量的维度顺序
- 将qkv分解为q、k、v
- 计算q和位置编码的乘积
- 组合相似性矩阵并归一化
- 加权求和得到v的注意力输出

## 2. 核心分析
### 类定义与参数
#### `class qkv_transform`
- **描述**: 用于qkv变换的Conv1d

#### `class AxialAttention`
- **描述**: 无文档说明。
- **初始化参数**: `in_planes, out_planes, groups, kernel_size, stride, bias, width`

## 3. 使用示例
```python
# 导入方式（参考）：from AxialAttention import ...

input = torch.randn(1, 64, 224, 224)  # 创建一个测试输入，大小为 1x64x224x224
    block = AxialAttention(in_planes=64, out_planes=64, groups=1, kernel_size=224, stride=1, bias=False, width=False)
    # in_planes、out_planes和channel一致，kernel_size和h,w一致
    output = block(input)
    print(input.size())  # 输出输入张量的尺寸
    print(output.size())  # 输出输出张量的尺寸
```

## 4. 适用任务
- **语义分割/实例分割**
- **医学图像处理**
- **Transformer相关任务**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
