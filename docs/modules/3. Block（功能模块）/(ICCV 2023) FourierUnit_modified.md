# Rethinking Fast Fourier Convolution in Image Inpainting

## 1. 模块简介
- **论文地址**: [https://openaccess.thecvf.com/content/ICCV2023/papers/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.pdf)
- **源文件**: `(ICCV 2023) FourierUnit_modified.py`

### 设计机制
- 中文题目：图像修复中快速傅里叶卷积的再思考
- 官方github：https://github.com/1911cty/Unbiased-Fast-Fourier-Convolution
- 所属机构：浙江大学计算机科学与技术学院，浙江工商大学
- bn_layer not used
- REPEAT CONV

## 2. 核心分析
### 类定义与参数
#### `class FourierUnit_modified`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, groups, spatial_scale_factor, spatial_scale_mode, spectral_pos_encoding, use_se, ffc3d, fft_norm`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICCV 2023) FourierUnit_modified import ...

# 实例化 FourierUnit_modified 模块
    model = FourierUnit_modified(
        in_channels=64,
        out_channels=64,
        groups=1,
        spatial_scale_factor=None,
        spectral_pos_encoding=False,
        use_se=False,
        ffc3d=False,
        fft_norm='ortho'
    )
    # 打印模型结构
    print(model)
    # 输入张量
    x = torch.randn(1, 64, 32,32)
    output = model(x)
    # 打印输入和输出的形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
```

## 4. 适用任务
- **通用视觉任务**: 图像分类、目标检测、语义分割等。
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
