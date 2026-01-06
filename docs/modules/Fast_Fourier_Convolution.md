# Fast_Fourier_Convolution

## 1. 模块简介
- **源文件**: `Fast_Fourier_Convolution.py`

### 设计机制
- ĿFast Fourier Convolution
- ĿٸҶ
- ӣhttps://proceedings.neurips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
- ٷgithub
- ѡоѧݿѧ
- һΪٸҶFast Fourier Convolution, FFC;
- ּʵַǾֲҰͿ߶ںϣڴͼƵʱܡ
- ΢ŹںšAI
- иҶ任ظ
- ֽͨʵ鲿ƴ
- ʵ鲿·
- 渵Ҷ任
- bn_layer not used
- groups_g = 1 if groups == 1 else int(groups * ratio_gout)
- groups_l = 1 if groups == 1 else groups - groups_g

## 2. 核心分析
### 类定义与参数
#### `class FourierUnit`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, groups`

#### `class SpectralTransform`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, stride, groups, enable_lfu`

#### `class FFC`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias, enable_lfu`

#### `class FFC_BN_ACT`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride, padding, dilation, groups, bias, norm_layer, activation_layer, enable_lfu`

## 3. 使用示例
```python
# 导入方式（参考）：from Fast_Fourier_Convolution import ...

print("ǰϵͳʱ:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 汾: {torch.__version__}")
    print(f"CUDA 汾: {torch.version.cuda}")
    print(f"CUDA Ƿ: {torch.cuda.is_available()}")
    print("΢Źں:AI,The test was successful!")

    ffc = FFC_BN_ACT(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1).to(device) 
    input = torch.rand(1, 16, 128, 128).to(device)  # 
    output = ffc(input)  # ǰ򴫲
    print(f"\n״: {input.shape}")
    print(f"״: {output.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
