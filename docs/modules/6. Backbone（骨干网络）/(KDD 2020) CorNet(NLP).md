# Correlation Networks for Extreme Multi-label Text Classification

## 1. 模块简介
- **源文件**: `(KDD 2020) CorNet(NLP).py`

### 设计机制
- 打印输入和输出的尺寸

## 2. 核心分析
### 类定义与参数
#### `class CorNetBlock`
- **描述**: 无文档说明。
- **初始化参数**: `context_size, output_size, cornet_act`

#### `class CorNet`
- **描述**: 无文档说明。
- **初始化参数**: `output_size, cornet_dim, n_cornet_blocks`

## 3. 使用示例
```python
# 导入方式（参考）：from (KDD 2020) CorNet(NLP) import ...

output_size = 10
    cornet_dim = 100
    n_cornet_blocks = 2
    cornet_act = 'relu'

    model = CorNet(output_size=output_size, cornet_dim=cornet_dim, n_cornet_blocks=n_cornet_blocks)

    input_tensor = torch.rand(4, output_size)

    output = model(input_tensor)

    # 打印输入和输出的尺寸
    print("Input size :", input_tensor.size())
    print("Output size:", output.size())
```

## 4. 适用任务
- **图像分类**
- **通用骨干网络/特征提取**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
