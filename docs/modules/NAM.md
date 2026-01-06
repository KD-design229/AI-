# NAM

## 1. 模块简介
- **源文件**: `NAM.py`

### 设计机制
- from torchinfo import summary  # ע
- from fvcore.nn import FlopCountAnalysis, flop_count_table  # ע
- ĿNAM: Normalization-based Attention Module
- Ŀ:  NAM ڹһעģ
- ӣhttps://arxiv.org/pdf/2111.12419
- ٷgithubhttps://github.com/Christian-lyc/NAM
- ѧҽѧԺϢѧԺ
- ؼʣһעռעͨעͼ
- ΢ŹںţAI
- עδԴռע룬´ɡ΢ŹںţAIṩ.
- x״Ϊ (B, C, H, W)
- ÿصȨ (Pixel Normalization)
- # 㣬ע
- print("Model Summary:")
- print(summary(nam, input_size=(1, 32, 256, 256), device=device.type))
- # 㣬ע
- flops = FlopCountAnalysis(nam, input)
- print("\nFlop Count Table:")
- print(flop_count_table(flops))

## 2. 核心分析
### 类定义与参数
#### `class Channel_Att`
- **描述**: 无文档说明。
- **初始化参数**: `channels, t`

#### `class Spatial_Att`
- **描述**: 无文档说明。
- **初始化参数**: `kernel_size`

#### `class NAM`
- **描述**: 无文档说明。
- **初始化参数**: `channels`

## 3. 使用示例
```python
# 导入方式（参考）：from NAM import ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nam = NAM(channels=32).to(device)  # ģƶ豸
    input = torch.rand(1, 32, 256, 256).to(device)  # ƶ豸

    # # 㣬ע
    # print("Model Summary:")
    # print(summary(nam, input_size=(1, 32, 256, 256), device=device.type))

    # # 㣬ע
    # flops = FlopCountAnalysis(nam, input)
    # print("\nFlop Count Table:")
    # print(flop_count_table(flops))

    output = nam(input)
    print(f"\nInput shape: {input.shape}")
    print(f"Output shape: {output.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
