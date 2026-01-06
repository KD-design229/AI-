# Attention Is All You Need

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)
- **源文件**: `(arXiv 2023) ScaledDotProductAttention.py`

## 2. 核心分析
### 类定义与参数
#### `class ScaledDotProductAttention`
- **描述**: Scaled dot-product attention
- **初始化参数**: `d_model, d_k, d_v, h, dropout`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2023) ScaledDotProductAttention import ...

input=torch.randn(50,49,512)
    sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
    output=sa(input,input,input)
    print(output.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
