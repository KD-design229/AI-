# (ICML 2024) RepBN

## 1. 模块简介
- **源文件**: `(ICML 2024) RepBN.py`

### 设计机制
- ĿSLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization
- ĿSLABмעͽʽزһĸЧ任
- ӣhttps://arxiv.org/pdf/2405.11582
- ٷgithubhttps://github.com/xinghaochen/SLAB
- ΪŵǷʵ
- :΢Źں:AI
- Դ, ά
- չά, :΢Źں:AI
- BatchNorm2dά
- if __name__ == "__main__":
- # ģ
- batch_size = 1    # С
- channels = 32     # ͨ
- N = 16 * 16      # ͼ߶* height * width
- model = RepBN(channels = channels)
- print(model)
- print("΢Źں:AI, nb!")
- #  (batch_size, channels, height * width (N))
- x = torch.randn(batch_size, N, channels)
- # ӡ״
- print("Input shape:", x.shape)
- # ǰ򴫲
- output = model(x)
- # ӡ״
- print("Output shape:", output.shape)
- (batch_size, channels, height, width)

## 2. 核心分析
### 类定义与参数
#### `class RepBN`
- **描述**: 无文档说明。
- **初始化参数**: `channels`

#### `class RepBN2d`
- **描述**: 无文档说明。
- **初始化参数**: `channels`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICML 2024) RepBN import ...

#     # ģ
#     batch_size = 1    # С
#     channels = 32     # ͨ
#     N = 16 * 16      # ͼ߶* height * width

#     model = RepBN(channels = channels)
#     print(model)
#     print("΢Źں:AI, nb!")

#     #  (batch_size, channels, height * width (N))
#     x = torch.randn(batch_size, N, channels)
#     # ӡ״
#     print("Input shape:", x.shape)
#     # ǰ򴫲
#     output = model(x)
#     # ӡ״
#     print("Output shape:", output.shape)

if __name__ == "__main__":
    # ģ
    batch_size = 1    # С
    channels = 32     # ͨ
    height = 256      # ͼ߶
    width = 256        # ͼ

    model = RepBN2d(channels = channels)
    print(model)
    print("΢Źں:AI, nb!")

    #  (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # ӡ״
    print("Input shape:", x.shape)
    # ǰ򴫲
    output = model(x)
    # ӡ״
    print("Output shape:", output.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
