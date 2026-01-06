# (TGRS 2024) EFC

## 1. 模块简介
- **源文件**: `(TGRS 2024) EFC.py`

### 设计机制
- ĿA Lightweight Fusion Strategy With Enhanced Interlayer Feature Correlation for Small Object Detection
- Ŀ:  ںϲǿԣСĿ
- ӣhttps://ieeexplore.ieee.org/abstract/document/10671587
- ٷgithubhttps://github.com/nuliweixiao/EFC
- ؼʣںϡСĿ
- ΢ŹںţAI

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

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
