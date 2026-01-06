# Polarized Self-Attention: Towards High-quality Pixel-wise Regression

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2107.00782](https://arxiv.org/pdf/2107.00782)
- **源文件**: `ParallelPolarizedSelfAttention.py`

### 设计机制
- 中文题目:  极化自注意力：面向高质量像素级回归
- 官方github：https://github.com/DeLightCMU/PSA
- 所属机构：南京理工大学, 卡内基梅隆大学
- 关键词： 极化自注意力, 像素级回归, 自注意力机制, 深度卷积神经网络, 长距离依赖
- 公众号：AI缝合术
- Channel-only Self-Attention
- Spatial-only Self-Attention

## 2. 核心分析
### 类定义与参数
#### `class ParallelPolarizedSelfAttention`
- **描述**: 无文档说明。
- **初始化参数**: `channel`

## 3. 使用示例
```python
# 导入方式（参考）：from ParallelPolarizedSelfAttention import ...

input=torch.randn(1,32,256,256)
    psa = ParallelPolarizedSelfAttention(channel=32)
    output=psa(input)
    print(output.shape)
```

## 4. 适用任务
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
