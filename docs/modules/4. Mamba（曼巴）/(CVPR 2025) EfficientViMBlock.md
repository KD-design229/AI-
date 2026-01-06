# EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2411.15241](https://arxiv.org/pdf/2411.15241)
- **源文件**: `(CVPR 2025) EfficientViMBlock.py`

### 设计机制
- 中文题目：高效ViM：基于隐藏状态混合器的状态空间对偶性的高效视觉Mamba
- 官方github：https://arxiv.org/pdf/2411.15241
- 所属机构：韩国大学——计算机科学与工程系

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

# 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建测试输入张量
    x = torch.randn(1, 32, 256, 256).to(device)

    # 初始化 evim 模块
    evim = EfficientViMBlock(dim=32).to(device)
    print(evim)

    # 前向传播    
    print("\n微信公众号: AI缝合术!\n")
    output = evim(x)
    
    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)
```

## 4. 适用任务
- **长序列建模/Mamba应用**
- **Mamba/长序列建模**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
