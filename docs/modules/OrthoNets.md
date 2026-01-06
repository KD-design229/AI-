# OrthoNets

## 1. 模块简介
- **源文件**: `OrthoNets.py`

### 设计机制
- ĿOrthoNets : Orthogonal Channel Attention Networks
- ĿOrthoNets : ͨע
- ӣhttps://arxiv.org/pdf/2311.03071
- ٷgithubhttps://github.com/hady1011/OrthoNets
- ɫѧѧϵ
- ΢ŹںšAI
- ע´ԴŻõԴ߼иʵܣԴΪɢңϸԴ
- class Orthogonal_Channel_Attention(torch.nn.Module):
- def __init__(self, c: int, h:int):
- super().__init__()
- self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
- self.FWT = GramSchmidtTransform.build(c, h)  # ʼ FWT
- def forward(self, input: Tensor):
- x = input
- while input[0].size(-1) > 1:
- input = self.FWT(input.to(self.device))
- b = input.size(0)
- return input.view(b, -1)
- Gram-Schmidt 任ʼ
- ͨעӳ䣨SE Block ṹ
- H  W ʼƥ䣬Ӧػ

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
    # ʼ Orthogonal_Channel_Attention
    channels = 32
    height = 256
    attention_module = Orthogonal_Channel_Attention(channels, height).to(device)
    #  (B, C, H, W)
    input_tensor = torch.rand(1, channels, 256, 256).to(device)
    # ǰ򴫲
    output_tensor = attention_module(input_tensor)
    print(f"״: {input_tensor.shape}")
    print(f"״: {output_tensor.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
