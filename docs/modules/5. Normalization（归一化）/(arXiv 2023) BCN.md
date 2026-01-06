# BCN: Batch Channel Normalization for Image Classification

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2312.00596](https://arxiv.org/pdf/2312.00596)
- **源文件**: `(arXiv 2023) BCN.py`

### 设计机制
- define parameters gamma, beta which are learnable
- dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
- initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
- define parameters running mean and variance which is not learnable
- keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
- variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
- calculate mean and variance along the dimensions other than the channel dimension
- variance calculation is using the biased formula during training
- during testing just use the running mean and (UnBiased) variance
- define parameters gamma, beta which are learnable
- dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
- initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
- define parameters running mean and variance which is not learnable
- keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
- variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)

## 2. 核心分析
### 类定义与参数
#### `class BatchNorm2D`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon, momentum, rescale`

#### `class BatchNormm2D`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon, momentum, rescale`

#### `class BatchNormm2DViiT`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon, momentum, rescale`

#### `class BatchNormm2DViTC`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon, momentum, rescale`

#### `class InstanceNorm2D`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon, momentum, rescale`

#### `class LayerNormViT`
- **描述**: 无文档说明。
- **初始化参数**: `features, eps`

#### `class LayerNormViTC`
- **描述**: 无文档说明。
- **初始化参数**: `features, eps`

#### `class LayerNorm2D`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon`

#### `class LayerNormm2D`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon`

#### `class GroupNorm2D`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, num_groups, epsilon`

#### `class BatchNorm_ByoL`
- **描述**: 无文档说明。
- **初始化参数**: `bn, num_channels, epsilon, momentum, rescale`

#### `class LaychNorm_ByoL`
- **描述**: 无文档说明。
- **初始化参数**: `bn, num_channels, epsilon, momentum, rescale`

#### `class BatchNorm_Byol`
- **描述**: 无文档说明。
- **初始化参数**: `bn, num_channels, epsilon, momentum, rescale`

#### `class LaychNorm_Byol`
- **描述**: 无文档说明。
- **初始化参数**: `bn, num_channels, epsilon, momentum, rescale`

#### `class BatchChannelNorm_Byol`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon, momentum`

#### `class BatchChannelNorm`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon, momentum`

#### `class BatchChannelNormvit`
- **描述**: 无文档说明。
- **初始化参数**: `num_channels, epsilon, momentum`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2023) BCN import ...

block = BatchChannelNorm(num_channels=64)
    input = torch.rand(64, 64, 9, 9)
    output = block(input)
    print(input.size())
    print(output.size())
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
- **归一化/模型稳定化**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
