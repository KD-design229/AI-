# (CVPR 2025) EfficientViMBlock

## 1. 模块简介
- **源文件**: `(CVPR 2025) EfficientViMBlock.py`

### 设计机制
- ĿEfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality
- ĿЧViM״̬״̬ռżԵĸЧӾMamba
- ӣhttps://arxiv.org/pdf/2411.15241
- ٷgithubhttps://arxiv.org/pdf/2411.15241
- ѧѧ빤ϵ
- ΢ŹںţAI

## 2. 核心分析
### 类定义与参数
#### `class LayerNorm2D`
- **描述**: LayerNorm for channels of 2D tensor(B C H W)
- **初始化参数**: `num_channels, eps, affine`

#### `class LayerNorm1D`
- **描述**: LayerNorm for channels of 1D tensor(B C L)
- **初始化参数**: `num_channels, eps, affine`

#### `class ConvLayer2D`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim, out_dim, kernel_size, stride, padding, dilation, groups, norm, act_layer, bn_weight_init`

#### `class ConvLayer1D`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim, out_dim, kernel_size, stride, padding, dilation, groups, norm, act_layer, bn_weight_init`

#### `class FFN`
- **描述**: 无文档说明。
- **初始化参数**: `in_dim, dim`

#### `class HSMSSD`
- **描述**: 无文档说明。
- **初始化参数**: `d_model, ssd_expand, A_init_range, state_dim`

#### `class EfficientViMBlock`
- **描述**: 无文档说明。
- **初始化参数**: `dim, mlp_ratio, ssd_expand, state_dim`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2025) EfficientViMBlock import ...

# ģƶ GPUã
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 
    x = torch.randn(1, 32, 256, 256).to(device)

    # ʼ evim ģ
    evim = EfficientViMBlock(dim=32).to(device)
    print(evim)

    # ǰ򴫲    
    print("\n΢Źں: AI!\n")
    output = evim(x)
    
    # ӡ״
    print("״:", x.shape)
    print("״:", output.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
