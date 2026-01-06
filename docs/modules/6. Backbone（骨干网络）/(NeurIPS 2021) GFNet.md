# Global Filter Networks for Image Classification

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2107.00645](https://arxiv.org/pdf/2107.00645)
- **源文件**: `(NeurIPS 2021) GFNet.py`

### 设计机制
- FIXME look at relaxing size constraints

## 2. 核心分析
### 类定义与参数
#### `class PatchEmbed`
- **描述**: Image to Patch Embedding
    
- **初始化参数**: `img_size, patch_size, in_chans, embed_dim`

#### `class GlobalFilter`
- **描述**: 无文档说明。
- **初始化参数**: `dim, h, w`

#### `class Mlp`
- **描述**: 无文档说明。
- **初始化参数**: `in_features, hidden_features, out_features, act_layer, drop`

#### `class Block`
- **描述**: 无文档说明。
- **初始化参数**: `dim, mlp_ratio, drop, drop_path, act_layer, norm_layer, h, w`

#### `class GFNet`
- **描述**: 无文档说明。
- **初始化参数**: `embed_dim, img_size, patch_size, mlp_ratio, depth, num_classes`

## 3. 使用示例
```python
# 导入方式（参考）：from (NeurIPS 2021) GFNet import ...

x = torch.randn(1, 3, 224, 224)
    gfnet = GFNet(embed_dim=384, img_size=224, patch_size=16, num_classes=1000)
    out = gfnet(x)
    print(out.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
