# OrthoNets : Orthogonal Channel Attention Networks

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2311.03071](https://arxiv.org/pdf/2311.03071)
- **源文件**: `OrthoNets.py`

### 设计机制
- 中文题目：OrthoNets : 正交通道注意力网络
- 官方github：https://github.com/hady1011/OrthoNets
- 所属机构：阿肯色大学计算机科学与计算机工程系
- 注：以下代码由源码优化整理得到，比源码逻辑更清晰，运行更流畅，且实现了完整功能，源码较为散乱，详细请见源码
- class Orthogonal_Channel_Attention(torch.nn.Module):
- def __init__(self, c: int, h:int):
- super().__init__()
- self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
- self.FWT = GramSchmidtTransform.build(c, h)  # 初始化 FWT
- def forward(self, input: Tensor):
- x = input
- while input[0].size(-1) > 1:
- input = self.FWT(input.to(self.device))
- b = input.size(0)
- return input.view(b, -1)

## 2. 核心分析
### 类定义与参数
#### `class GramSchmidtTransform`
- **描述**: 无文档说明。
- **初始化参数**: `c, h`

#### `class Orthogonal_Channel_Attention`
- **描述**: 无文档说明。
- **初始化参数**: `channels, height`

## 3. 使用示例
```python
# 导入方式（参考）：from OrthoNets import ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化 Orthogonal_Channel_Attention
    channels = 32
    height = 256
    attention_module = Orthogonal_Channel_Attention(channels, height).to(device)
    # 输入张量 (B, C, H, W)
    input_tensor = torch.rand(1, channels, 256, 256).to(device)
    # 前向传播
    output_tensor = attention_module(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
```

## 4. 适用任务
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
