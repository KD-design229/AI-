# Frequency-aware Feature Fusion for Dense Image Prediction

## 1. 模块简介
- **论文地址**: [https://github.com/Linwei-Chen/FreqFusion](https://github.com/Linwei-Chen/FreqFusion)
- **源文件**: `(TPAMI 2024) FreqFusion.py`

### 设计机制
- 公众号：AI缝合术
- 生成水平和垂直方向上的Hamming窗
- hamming_x = np.blackman(M)
- hamming_x = np.kaiser(M)
- 通过外积生成二维Hamming窗

## 2. 核心分析
### 类定义与参数
#### `class FreqFusion`
- **描述**: 无文档说明。
- **初始化参数**: `hr_channels, lr_channels, scale_factor, lowpass_kernel, highpass_kernel, up_group, encoder_kernel, encoder_dilation, compressed_channels, align_corners, upsample_mode, feature_resample, feature_resample_group, comp_feat_upsample, use_high_pass, use_low_pass, hr_residual, semi_conv, hamming_window, feature_resample_norm`

#### `class LocalSimGuidedSampler`
- **描述**: offset generator in FreqFusion
- **初始化参数**: `in_channels, scale, style, groups, use_direct_scale, kernel_size, local_window, sim_type, norm, direction_feat`

## 3. 使用示例
```python
# 导入方式（参考）：from (TPAMI 2024) FreqFusion import ...

# 设置设备为GPU（如果可用）否则为CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 初始化模型并移动到设备
    ff = FreqFusion(hr_channels=64, lr_channels=64).to(device)
    # 创建随机输入并移动到设备
    hr_feat = torch.rand(1, 64, 32, 32).to(device)
    lr_feat = torch.rand(1, 64, 16, 16).to(device)
    # 前向传播
    _, hr_feat, lr_feat = ff(hr_feat=hr_feat, lr_feat=lr_feat)  # lr_feat [1, 64, 32, 32]
    # 打印输出的形状
    print(hr_feat.shape)
    print(lr_feat.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
