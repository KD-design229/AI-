# classification

## 1. 模块简介
- **论文地址**: [https://github.com/Ray010221/MCANet](https://github.com/Ray010221/MCANet)
- **源文件**: `MCAM.py`

## 2. 核心分析
### 类定义与参数
#### `class MCAM`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, inter_channels, dimension, sub_sample, bn_layer`

## 3. 使用示例
```python
# 导入方式（参考）：from MCAM import ...

block = MCAM(in_channels=256)
    sar = torch.randn(2, 256, 64, 64)
    opt = torch.randn(2, 256, 64, 64)
    print("input:", sar.shape, opt.shape)
    print("output:", block(sar, opt).shape)
```

## 4. 适用任务
- **语义分割/实例分割**
- **图像分类**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
