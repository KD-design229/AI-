# Strip Pooling: Rethinking spatial pooling for scene parsing  (CVPR2020)

## 1. 模块简介
- **论文地址**: [https://github.com/houqb/SPNet](https://github.com/houqb/SPNet)
- **源文件**: `(CVPR 2020) strip_pooling.py`

### 设计机制
- ---------------------------------------
- ---------------------------------------
- bilinear interpolate options
- 输入 B C H W,  输出 B C H W

## 2. 核心分析
### 类定义与参数
#### `class StripPooling`
- **描述**: Reference:
- **初始化参数**: `in_channels, pool_size, norm_layer, up_kwargs`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2020) strip_pooling import ...

block = StripPooling(64, (20, 12), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})
    input = torch.rand(3, 64, 32, 32)
    output = block(input)
    print(input.size(), output.size())
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
- **特征采样/尺度变换**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
