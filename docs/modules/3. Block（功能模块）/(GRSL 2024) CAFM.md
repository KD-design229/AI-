# (GRSL 2024) CAFM

## 1. 模块简介
- **源文件**: `(GRSL 2024) CAFM.py`

### 设计机制
- ĿHybrid Convolutional and Attention Network for Hyperspectral Image Denoising
- Ŀ:  Ͼעڸ߹ͼȥ
- ӣhttps://arxiv.org/pdf/2403.10067
- ٷgithubhttps://github.com/summitgao/HCANet
- йѧѧ뼼ѧԺѧϵ
- ؼʣͼͼȥ룬任עƣѧϰ
- ΢ŹںšAI
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

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
