# (ECCV 2024) FFCM

## 1. 模块简介
- **源文件**: `(ECCV 2024) FFCM.py`

### 设计机制
- ĿEfficient Frequency-Domain Image Deraining with Contrastive Regularization
- ĿЧƵͼȥԱȶ
- ӣhttps://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
- ٷgithubhttps://github.com/deng-ai-lab/FADformer
- պѧѧԺ
- ؼʣSIDƵѧϰԱ
- ΢ŹںšAI
- (batch, c, h, w/2+1, 2)
- (batch, c, 2, h, w/2+1)

## 2. 核心分析
### 类定义与参数
#### `class FourierUnit`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, groups`

#### `class Freq_Fusion`
- **描述**: 无文档说明。
- **初始化参数**: `dim, kernel_size, se_ratio, local_size, scale_ratio, spilt_num`

#### `class Fused_Fourier_Conv_Mixer`
- **描述**: 无文档说明。
- **初始化参数**: `dim, token_mixer_for_gloal, mixer_kernel_size, local_size`

## 3. 使用示例
```python
# 导入方式（参考）：from (ECCV 2024) FFCM import ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ffcm = Fused_Fourier_Conv_Mixer(32).to(device) 
    input = torch.rand(1, 32, 256, 256).to(device)
    output = ffcm(input)
    
    print(f"\nInput shape: {input.shape}")
    print(f"Output shape: {output.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
