# ParallelPolarizedSelfAttention

## 1. 模块简介
- **源文件**: `ParallelPolarizedSelfAttention.py`

### 设计机制
- ĿPolarized Self-Attention: Towards High-quality Pixel-wise Regression
- Ŀ:  עؼع
- ӣhttps://arxiv.org/pdf/2107.00782
- ٷgithubhttps://github.com/DeLightCMU/PSA
- Ͼѧ, ڻ÷¡ѧ
- ؼʣ ע, ؼع, ע, Ⱦ,
- ںţAI
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

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
