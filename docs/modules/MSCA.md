# FcaNet: Frequency Channel Attention Networks

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2012.11879](https://arxiv.org/pdf/2012.11879)
- **源文件**: `MSCA.py`

### 设计机制
- 中文题目:  FcaNet: 频域通道注意力网络
- 官方github：https://github.com/cfzd/FcaNet
- 所属机构：浙江大学计算机学院，浙江大学上海高等研究院
- 关键词：频域，通道注意力，图像分类，目标检测，语义分割
- Ensure that frequencies of different sizes have the same representation in the identical 7x7 frequency space.
- make the frequencies in different sizes are identical to a 7x7 frequency space
- eg, (2,2) in 14x14 is identical to (1,1) in 7x7
- multi-spectral information aggregate
- If you have concerns about one-line-change, don't worry.   :)
- In the ImageNet models, this line will never be triggered.
- This is for compatibility in instance segmentation and object detection.

## 2. 核心分析
### 类定义与参数
#### `class MultiSpectralAttentionLayer`
- **描述**: the implementation of FCA
- **初始化参数**: `channel, dct_h, dct_w, reduction, freq_sel_method`

#### `class MultiSpectralDCTLayer`
- **描述**: Generate dct filters
- **初始化参数**: `height, width, mapper_x, mapper_y, channel`

## 3. 使用示例
```python
# 导入方式（参考）：from MSCA import ...

x       = torch.randn(1,256,128,128)
    model   = MultiSpectralAttentionLayer(256,128,128)
    print(model(x).shape)
```

## 4. 适用任务
- **语义分割/实例分割**
- **目标检测**
- **图像分类**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
