# (CVPR 2023) FSAS

## 1. 模块简介
- **源文件**: `(CVPR 2023) FSAS.py`

### 设计机制
- ĿEfficient Frequency Domain-based Transformers for High-Quality Image Deblurring
- ĿƵĸЧTransformerڸͼȥģ
- ӣhttps://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf
- ٷgithubhttps://github.com/kkkls/FFTformer
- Ͼѧѧ빤ѧԺйӿƼϢѧоԺ
- ΢ŹںšAI

## 2. 核心分析
### 类定义与参数
#### `class BiasFree_LayerNorm`
- **描述**: 无文档说明。
- **初始化参数**: `normalized_shape`

#### `class WithBias_LayerNorm`
- **描述**: 无文档说明。
- **初始化参数**: `normalized_shape`

#### `class LayerNorm`
- **描述**: 无文档说明。
- **初始化参数**: `dim, LayerNorm_type`

#### `class FSAS`
- **描述**: 无文档说明。
- **初始化参数**: `dim, bias`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2023) FSAS import ...

####ע####
    print("ǰϵͳʱ:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 汾: {torch.__version__}")
    print(f"CUDA 汾: {torch.version.cuda}")
    print(f"CUDA Ƿ: {torch.cuda.is_available()}")
    print("΢Źں:AI,The test was successful!")
    ####ע####

    fsas= FSAS(32).to(device)
    input = torch.rand(1, 32, 256, 256).to(device)  # 
    output = fsas(input)  # ǰ򴫲
    print(f"\n״: {input.shape}")
    print(f"״: {output.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
