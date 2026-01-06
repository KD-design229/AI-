# (PR2023) HaarDownsampling

## 1. 模块简介
- **论文地址**: [https://www.sciencedirect.com/science/article/pii/S0031320323005174](https://www.sciencedirect.com/science/article/pii/S0031320323005174)
- **源文件**: `(PR2023) HaarDownsampling.py`

### 设计机制
- 运行报错ModuleNotFoundError: No module named 'pytorch_wavelets'
- 说明缺少pytorch_wavelets包, 用 pip install pytorch-wavelets 即可安装
- 初始化离散小波变换，J=1表示变换的层数，mode='zero'表示填充模式，使用'Haar'小波
- 定义卷积、批归一化和ReLU激活的顺序组合
- 对输入x进行离散小波变换，得到低频部分yL和高频部分yH
- 提取高频部分的不同分量
- 将低频部分和高频部分拼接在一起
- 通过卷积、批归一化和ReLU激活处理拼接后的特征

## 2. 核心分析
### 类定义与参数
#### `class Down_wt`
- **描述**: 无文档说明。
- **初始化参数**: `in_ch, out_ch`

## 3. 使用示例
```python
# 导入方式（参考）：from (PR2023) HaarDownsampling import ...

block = Down_wt(64, 64)  # 创建 Down_wt 模块，输入和输出通道数均为64
    input = torch.rand(3, 64, 64, 64)  # 创建输入张量，形状为 (B, C, H, W)
    output = block(input)  # 通过模块处理输入
    print(input.size())  # 打印输入的尺寸
    print(output.size())  # 打印输出的尺寸
```

## 4. 适用任务
- **目标检测**
- **图像分类**
- **图像去噪**
- **目标跟踪**
- **去雨**
- **去雾**
- **去模糊**
- **图像融合**
- **语义分割/实例分割**
- **超分辨率**
- **特征采样/尺度变换**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
