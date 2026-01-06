# SCTNet: Single-Branch CNN with Transformer Semantic Information for Real-Time Segmentation

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2312.17071](https://arxiv.org/pdf/2312.17071)
- **源文件**: `(AAAI 2024) CFBlock.py`

### 设计机制
- 中文题目: 单分支CNN结合Transformer语义信息的实时分割网络
- 官方github：https://github.com/xzz777/SCTNet
- 所属机构：华中科技大学人工智能与自动化学院国家多媒体信息智能处理技术重点实验室
- 关键词：实时语义分割，Transformer，单分支CNN，语义信息对齐，深度学习
- BN->Conv->GELU->drop->Conv2->drop

## 2. 核心分析
### 类定义与参数
#### `class MLP`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, hidden_channels, out_channels, drop_rate`

#### `class ConvolutionalAttention`
- **描述**: The ConvolutionalAttention implementation
Args:
    in_channels (int, optional): The input channels.
    inter_channels (int, optional): The channels of intermediate feature.
    out_channels (int, optional): The output channels.
    num_heads (int, optional): The num of heads in attention. Default: 8
- **初始化参数**: `in_channels, out_channels, inter_channels, num_heads`

#### `class CFBlock`
- **描述**: The CFBlock implementation based on PaddlePaddle.
Args:
    in_channels (int, optional): The input channels.
    out_channels (int, optional): The output channels.
    num_heads (int, optional): The num of heads in attention. Default: 8
    drop_rate (float, optional): The drop rate in MLP. Default:0.
    drop_path_rate (float, optional): The drop path rate in CFBlock. Default: 0.2
- **初始化参数**: `in_channels, out_channels, num_heads, drop_rate, drop_path_rate`

## 3. 使用示例
```python
# 导入方式（参考）：from (AAAI 2024) CFBlock import ...

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input=torch.randn(1,32,256,256).to(device)
    print(input.shape)
    cfb = CFBlock(32,32).to(device)
    output=cfb(input)
    print(output.shape)
```

## 4. 适用任务
- **语义分割/实例分割**
- **Transformer相关任务**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
