# Hybrid Convolutional and Attention Network for Hyperspectral Image Denoising

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2403.10067](https://arxiv.org/pdf/2403.10067)
- **源文件**: `(GRSL 2024) CAFM.py`

### 设计机制
- 中文题目:  混合卷积和注意力网络用于高光谱图像去噪
- 官方github：https://github.com/summitgao/HCANet
- 所属机构：中国海洋大学计算机科学与技术学院，密西西比州立大学电气与计算机工程系
- 关键词：超光谱图像，图像去噪，变换器，注意力机制，深度学习
- local conv
- global SA

## 2. 核心分析
### 类定义与参数
#### `class CAFM`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_heads, bias`

## 3. 使用示例
```python
# 导入方式（参考）：from (GRSL 2024) CAFM import ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cafm = CAFM(dim=256, num_heads=8).to(device) 
    input = torch.rand(1, 256, 64, 64).to(device)
    output = cafm(input)
    
    print(f"\nInput shape: {input.shape}")
    print(f"Output shape: {output.shape}")
```

## 4. 适用任务
- **图像去噪**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
