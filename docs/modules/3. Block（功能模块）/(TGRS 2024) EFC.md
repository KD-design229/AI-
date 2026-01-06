# A Lightweight Fusion Strategy With Enhanced Interlayer Feature Correlation for Small Object Detection

## 1. 模块简介
- **论文地址**: [https://ieeexplore.ieee.org/abstract/document/10671587](https://ieeexplore.ieee.org/abstract/document/10671587)
- **源文件**: `(TGRS 2024) EFC.py`

### 设计机制
- 中文题目:  轻量级融合策略增强层间特征相关性，用于小目标检测
- 官方github：https://github.com/nuliweixiao/EFC
- 所属机构：北京理工大学
- 关键词：特征融合、轻量级、小目标检测

## 2. 核心分析
### 类定义与参数
#### `class EFC`
- **描述**: 无文档说明。
- **初始化参数**: `c1, c2`

## 3. 使用示例
```python
# 导入方式（参考）：from (TGRS 2024) EFC import ...

x1 = torch.randn(1,32,256,256)
    x2 = torch.randn(1,32,256,256)
    x = (x1,x2)
    model = EFC(32,32)
    print(model(x).shape)
```

## 4. 适用任务
- **目标检测**
- **图像融合**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
