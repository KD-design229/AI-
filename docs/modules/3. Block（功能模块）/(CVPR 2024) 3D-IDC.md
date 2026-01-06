# InceptionNeXt: When Inception Meets ConvNeXt (CVPR 2024)

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2303.16900](https://arxiv.org/pdf/2303.16900)
- **源文件**: `(CVPR 2024) 3D-IDC.py`

## 2. 核心分析
### 类定义与参数
#### `class InceptionDWConv3d`
- **描述**: Inception depthwise convolution for 3D data
    
- **初始化参数**: `in_channels, cube_kernel_size, band_kernel_size, branch_ratio`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2024) 3D-IDC import ...

block = InceptionDWConv3d(64) # 输入 C
    input = torch.randn(1, 64, 16, 224, 224) # 输入B C D H W
    output = block(input)
    print(input.size())
    print(output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
