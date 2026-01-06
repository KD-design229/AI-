# Encoder-Decoder Based Convolutional Neural Networks with Multi-Scale-Aware Modules for Crowd Counting

## 1. 模块简介
- **论文地址**: [https://ieeexplore.ieee.org/document/9413286](https://ieeexplore.ieee.org/document/9413286)
- **源文件**: `(ICPR 2021) CAN.py`

## 2. 核心分析
### 类定义与参数
#### `class ContextualModule`
- **描述**: 无文档说明。
- **初始化参数**: `features, out_features, sizes`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICPR 2021) CAN import ...

block = ContextualModule(features=64, out_features=64)
    input_tensor = torch.rand(1, 64, 128, 128)
    output = block(input_tensor)
    print("Input size:", input_tensor.size())
    print("Output size:", output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
