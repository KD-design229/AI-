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

## 4. 适用任务
- **目标检测**
- **图像分类**
- **图像去噪**
- **目标跟踪**
- **去雨**
- **去雾**
- **去模糊**
- **图像融合**
- **语义分割/实例分割**
- **超分辨率**
- **Transformer相关任务**
- **注意力机制应用**
- **通用骨干网络/特征提取**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
