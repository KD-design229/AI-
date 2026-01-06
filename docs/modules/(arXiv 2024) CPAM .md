# (arXiv 2024) CPAM 

## 1. 模块简介
- **源文件**: `(arXiv 2024) CPAM .py`

### 设计机制
- ĿASF-YOLO: A novel YOLO model with attentional scale sequence fusion for cell instance segmentation
- Ŀ:  ASF-YOLOעƵĳ߶ںYOLOϸͼʵָ
- ӣhttps://arxiv.org/pdf/2312.06458
- ٷgithubhttps://github.com/mkang315/ASF-YOLO
- ĪʲѧϢѧԺ
- ؼʣҽѧͼСָYOLOںϣע
- Channel and Position Attention Mechanism (CPAM)

## 2. 核心分析
### 类定义与参数
#### `class channel_att`
- **描述**: 无文档说明。
- **初始化参数**: `channel, b, gamma`

#### `class local_att`
- **描述**: 无文档说明。
- **初始化参数**: `channel, reduction`

#### `class CPAM`
- **描述**: 无文档说明。
- **初始化参数**: `ch`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2024) CPAM  import ...

cpam = CPAM(32)

    input1 = torch.randn(1, 32, 256, 256)
    input2 = torch.randn(1, 32, 256, 256)
    print(input1.size())
    print(input2.size())

    inputs = [input1, input2]

    output = cpam(inputs)

    print(output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
