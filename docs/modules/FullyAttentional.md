# FullyAttentional

## 1. 模块简介
- **源文件**: `FullyAttentional.py`

### 设计机制
- ĿFully Attentional Network for Semantic Segmentation
- ӣhttps://arxiv.org/pdf/2112.04108
- ٷgithubhttps://github.com/maggiesong7/FullyAttentional?tab=readme-ov-file
- עͣںţAI
- AIgithubhttps://github.com/AIFengheshu/Plug-play-modules
- https://github.com/Ilareina/FullyAttentional/blob/main/model.py
- ʼplaneͼͨnorm_layerǹһ㣨ĬΪBatchNorm2d
- ȫӲ㣬conv1conv2
- + һ + ReLU
- softmaxڼϵ
- ʼѧϰĲgammaڵյ
- ǰ򴫲̣xΪͼ״Ϊ (batch_size, channels, height, width)
- кͱΣȡˮƽʹֱ
- ֱˮƽʹֱгػͨȫӲб
- ˮƽʹֱĹϵ
- 㾭softmaxĹϵ
- ͨ˷͹ϵ󣬶мȨǿ
- ˮƽʹֱǿںϣԭʼ
- ͨнһ
- (B, C, H, W)

## 2. 核心分析
### 类定义与参数
#### `class FullyAttentionalBlock`
- **描述**: 无文档说明。
- **初始化参数**: `plane, norm_layer`

## 3. 使用示例
```python
# 导入方式（参考）：from FullyAttentional import ...

fab = FullyAttentionalBlock(plane=32).cuda()
    #  (B, C, H, W)
    input_tensor = torch.rand(1, 32, 256, 256).cuda()
    # ӡ״
    print(f"״: {input_tensor.shape}")
    # ǰ򴫲
    output_tensor = fab(input_tensor)
    # ӡ״
    print(f"״: {output_tensor.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
