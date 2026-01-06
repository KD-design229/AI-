# Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2106.12368](https://arxiv.org/pdf/2106.12368)
- **源文件**: `(TPAMI 2022) ViP.py`

## 2. 核心分析
### 类定义与参数
#### `class MLP`
- **描述**: 无文档说明。
- **初始化参数**: `in_features, hidden_features, out_features, act_layer, drop`

#### `class WeightedPermuteMLP`
- **描述**: 无文档说明。
- **初始化参数**: `dim, seg_dim, qkv_bias, proj_drop`

## 3. 使用示例
```python
# 导入方式（参考）：from (TPAMI 2022) ViP import ...

input=torch.randn(64,8,8,512)
    seg_dim=8
    vip=WeightedPermuteMLP(512,seg_dim)
    out=vip(input)
    print(out.shape)
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
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
