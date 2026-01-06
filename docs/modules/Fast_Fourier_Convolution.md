# Fast Fourier Convolution

## 1. 模块简介
- **论文地址**: [https://proceedings.neurips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)
- **源文件**: `Fast_Fourier_Convolution.py`

### 设计机制
- 中文题目：快速傅里叶卷积
- 官方github：无
- 所属机构：王选计算机技术研究所，北京大学数据科学中心
- 核心速览：本文提出了一种名为快速傅里叶卷积（Fast Fourier Convolution, FFC）的新型卷积操作符，
- 旨在实现非局部感受野和跨尺度融合，以提高深度网络在处理图像和视频任务时的性能。
- 进行傅里叶变换，返回复数结果
- 将复数分解成两个通道：实部和虚部拼接
- 将卷积后的实部和虚部重新分离
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

print("当前系统时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print("微信公众号:AI缝合术,The test was successful!")

    ffc = FFC_BN_ACT(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1).to(device) 
    input = torch.rand(1, 16, 128, 128).to(device)  # 输入张量
    output = ffc(input)  # 前向传播
    print(f"\n输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
```

## 4. 适用任务
- **语义分割/实例分割**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
