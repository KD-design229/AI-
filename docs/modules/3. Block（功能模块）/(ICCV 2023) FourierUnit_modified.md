# (ICCV 2023) FourierUnit_modified

## 1. 模块简介
- **源文件**: `(ICCV 2023) FourierUnit_modified.py`

### 设计机制
- ĿRethinking Fast Fourier Convolution in Image Inpainting
- Ŀͼ޸пٸҶ˼
- ӣhttps://openaccess.thecvf.com/content/ICCV2023/papers/Chu_Rethinking_Fast_Fourier_Convolution_in_Image_Inpainting_ICCV_2023_paper.pdf
- ٷgithubhttps://github.com/1911cty/Unbiased-Fast-Fourier-Convolution
- 㽭ѧѧ뼼ѧԺ㽭̴ѧ
- ΢ŹںšAI
- bn_layer not used
- REPEAT CONV
- irfft

## 2. 核心分析
### 类定义与参数
#### `class FourierUnit_modified`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, groups, spatial_scale_factor, spatial_scale_mode, spectral_pos_encoding, use_se, ffc3d, fft_norm`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICCV 2023) FourierUnit_modified import ...

# ʵ FourierUnit_modified ģ
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
    # ӡģͽṹ
    print(model)
    # 
    x = torch.randn(1, 64, 32,32)
    output = model(x)
    # ӡ״
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
