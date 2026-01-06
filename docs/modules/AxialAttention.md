# AxialAttention

## 1. 模块简介
- **源文件**: `AxialAttention.py`

### 设计机制
- ĿMedical Transformer: Gated Axial-Attention forMedical Image Segmentation
- Ŀ:  ҽTransformerҽѧͼָſע
- ӣhttps://arxiv.org/pdf/2102.10662
- ٷgithubhttps://github.com/jeya-maria-jose/Medical-Transformer
- Լս˹ѧ, ѧ
- ؼʣ Transformer, ҽѧͼָ, ע
- 1x1ڸıͨ
- һһά㣬ڽqkv任
- עģ飨Axial Attention
- ȷܱͨͨ
- ͷעqkv任
- 1ӳػ
- ݿȻ߶ȵά˳
- qkv任
- qkvֽΪqkv
- qλñĳ˻
- Ծ󲢹һ
- Ȩ͵õvע

## 2. 核心分析
### 类定义与参数
#### `class qkv_transform`
- **描述**: qkv任Conv1d

#### `class AxialAttention`
- **描述**: 无文档说明。
- **初始化参数**: `in_planes, out_planes, groups, kernel_size, stride, bias, width`

## 3. 使用示例
```python
# 导入方式（参考）：from AxialAttention import ...

input = torch.randn(1, 64, 224, 224)  # һ룬СΪ 1x64x224x224
    block = AxialAttention(in_planes=64, out_planes=64, groups=1, kernel_size=224, stride=1, bias=False, width=False)
    # in_planesout_planeschannelһ£kernel_sizeh,wһ
    output = block(input)
    print(input.size())  # ĳߴ
    print(output.size())  # ĳߴ
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
