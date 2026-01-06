# Spatial Group-wise Enhance: Improving Semantic

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/1905.09646](https://arxiv.org/pdf/1905.09646)
- **源文件**: `SGE.py`

### 设计机制
- Feature Learning in Convolutional Networks
- 中文题目:  空间分组增强：在卷积网络中改进语义特征学习
- 官方github：https://github.com/implus/PytorchInsight
- 所属机构：南京理工大学PCALab、Momenta、清华大学
- 关键词：卷积神经网络、注意力机制、图像分类、目标检测、特征增强

## 2. 核心分析
### 类定义与参数
#### `class SpatialGroupEnhance`
- **描述**: 无文档说明。
- **初始化参数**: `groups`

## 3. 使用示例
```python
# 导入方式（参考）：from SGE import ...

input=torch.randn(1,32,256,256)
    sge = SpatialGroupEnhance(groups=8)
    output=sge(input)
    print(output.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
