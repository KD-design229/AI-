# (AAAI 2024) CFBlock

## 1. 模块简介
- **源文件**: `(AAAI 2024) CFBlock.py`

### 设计机制
- ĿSCTNet: Single-Branch CNN with Transformer Semantic Information for Real-Time Segmentation
- Ŀ: ֧CNNTransformerϢʵʱָ
- ӣhttps://arxiv.org/pdf/2312.17071
- ٷgithubhttps://github.com/xzz777/SCTNet
- пƼѧ˹ԶѧԺҶýϢܴصʵ
- ؼʣʵʱָTransformer֧CNNϢ룬ѧϰ
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

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
