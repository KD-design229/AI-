# https://arxiv.org/pdf/2302.13800

## 1. 模块简介
- **源文件**: `(ICCV 2023) SAFM.py`

### 设计机制
- https://github.com/sunny2109/SAFMN
- Spatial Weighting
- # Feature Aggregation
- Activation

## 2. 核心分析
### 类定义与参数
#### `class SAFM`
- **描述**: 无文档说明。
- **初始化参数**: `dim, n_levels`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICCV 2023) SAFM import ...

input = torch.randn(3,36,64,64) #输入b c h w

    block = SAFM(dim=36)
    output =block(input)
    print(output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
