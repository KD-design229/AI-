# Momentum Contrast for Unsupervised Visual Representation Learning

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/1911.05722](https://arxiv.org/pdf/1911.05722)
- **源文件**: `(CVPR 2020) CBDE.py`

### 设计机制
- create the encoders
- num_classes is the output fc dimension
- create the queue
- gather keys before updating queue
- keys = concat_all_gather(keys)
- replace the keys at ptr (dequeue and enqueue)
- gather from all gpus
- random shuffle index
- broadcast to all gpus
- index for restoring
- shuffled index for this gpu

## 2. 核心分析
### 类定义与参数
#### `class MoCo`
- **描述**: Build a MoCo model with: a query encoder, a key encoder, and a queue
https://arxiv.org/abs/1911.05722
- **初始化参数**: `base_encoder, dim, K, m, T, mlp`

#### `class ResBlock`
- **描述**: 无文档说明。
- **初始化参数**: `in_feat, out_feat, stride`

#### `class ResEncoder`
- **描述**: 无文档说明。

#### `class CBDE`
- **描述**: 无文档说明。
- **初始化参数**: `opt`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2020) CBDE import ...

block = CBDE()
    input = torch.rand()
    output = block(input)
    print(input.size())
    print(output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
