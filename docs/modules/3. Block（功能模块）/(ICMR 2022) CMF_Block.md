# M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2104.09770](https://arxiv.org/pdf/2104.09770)
- **源文件**: `(ICMR 2022) CMF_Block.py`

## 2. 核心分析
### 类定义与参数
#### `class CMA_Block`
- **描述**: 无文档说明。
- **初始化参数**: `in_channel, hidden_channel, out_channel`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICMR 2022) CMF_Block import ...

in_channel = 64
    hidden_channel = 32
    out_channel = 64
    h = 64
    w = 64

    block = CMA_Block(in_channel, hidden_channel, out_channel)

    rgb_input = torch.rand(1, in_channel, h, w)
    freq_input = torch.rand(1, in_channel, h, w)

    output = block(rgb_input, freq_input)

    print("RGB Input size:", rgb_input.size())
    print("Freq Input size:", freq_input.size())
    print("Output size:", output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
