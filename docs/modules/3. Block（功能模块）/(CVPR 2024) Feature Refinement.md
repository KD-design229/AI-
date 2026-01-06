# Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration, CVPR 2024.

## 1. 模块简介
- **论文地址**: [https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)
- **源文件**: `(CVPR 2024) Feature Refinement.py`

### 设计机制
- bs x hw x c
- spatial restore
- gate mechanism
- Instantiate the FRFN class
- Create an instance of the FRFN module
- Generate a random input tensor
- Forward pass
- Print input and output shapes

## 2. 核心分析
### 类定义与参数
#### `class FRFN`
- **描述**: 无文档说明。
- **初始化参数**: `dim, hidden_dim, act_layer`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2024) Feature Refinement import ...

# Instantiate the FRFN class
    dim = 64  # Dimension of input features


    # Create an instance of the FRFN module
    frfn = FRFN(dim)

    # Generate a random input tensor
    B = 1  # Batch size
    H = 64  # Height of the feature map
    W = 64  # Width of the feature map
    C = dim  # Number of channels

    input = torch.randn(B, H * W, C)

    # Forward pass
    output = frfn(input)

    # Print input and output shapes
    print(input.size())
    print(output.size())
```

## 4. 适用任务
- **Transformer相关任务**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
