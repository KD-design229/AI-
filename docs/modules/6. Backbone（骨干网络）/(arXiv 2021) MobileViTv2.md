# Separable Self-attention for Mobile Vision Transformers

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2206.02680](https://arxiv.org/pdf/2206.02680)
- **源文件**: `(arXiv 2021) MobileViTv2.py`

## 2. 核心分析
### 类定义与参数
#### `class MobileViTv2Attention`
- **描述**: Scaled dot-product attention
- **初始化参数**: `d_model`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2021) MobileViTv2 import ...

input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
