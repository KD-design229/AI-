# Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring

## 1. 模块简介
- **论文地址**: [https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf)
- **源文件**: `(CVPR 2023) FSAS.py`

### 设计机制
- 中文题目：基于频域的高效Transformer用于高质量图像去模糊
- 官方github：https://github.com/kkkls/FFTformer
- 所属机构：南京理工大学计算机科学与工程学院，中国电子科技集团信息科学研究院

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

####可注释####
    print("当前系统时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print("微信公众号:AI缝合术,The test was successful!")
    ####可注释####

    fsas= FSAS(32).to(device)
    input = torch.rand(1, 32, 256, 256).to(device)  # 输入张量
    output = fsas(input)  # 前向传播
    print(f"\n输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")
```

## 4. 适用任务
- **去模糊**
- **Transformer相关任务**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
