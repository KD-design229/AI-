# (TGRS 2024) ELGCA

## 1. 模块简介
- **源文件**: `(TGRS 2024) ELGCA.py`

### 设计机制
- ĿELGC-Net: Efficient Local-Global Context Aggregation for Remote Sensing Change Detection
- ĿELGC-Netңб仯ĸЧֲ-ȫľۺ
- ӣhttps://arxiv.org/pdf/2403.17909
- ٷgithubhttps://github.com/techmn/elgcnet
- ºĬ¡Ү˹ܴѧIBM оĴǹѧѩƽѧ
- ΢ŹںšAI
- apply depth-wise convolution on half channels
- linear projection of other half before computing attention
- ģƶ GPUã
- (batch_size, channels, height, width)
- ʼ elgca ģ

## 2. 核心分析
### 类定义与参数
#### `class ELGCA`
- **描述**: Efficient local global context aggregation module
dim: number of channels of input
heads: number of heads utilized in computing attention
- **初始化参数**: `dim, heads`

## 3. 使用示例
```python
# 导入方式（参考）：from (TGRS 2024) ELGCA import ...

# ģƶ GPUã
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)

    # ʼ elgca ģ
    elgca = ELGCA(dim=32, heads=4)
    print(elgca)
    elgca = elgca.to(device)

    # ǰ򴫲
    output = elgca(x)

    # ӡ״
    print("״:", x.shape)
    print("״:", output.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
