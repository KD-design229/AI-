# (2024) MobileNetV4

## 1. 模块简介
- **源文件**: `(2024) MobileNetV4.py`

### 设计机制
- ĿMobileNetV4: Universal Models for the Mobile Ecosystem
- Ŀ: MobileNetV4ƶ̬ϵͳͨģ
- ӣhttps://arxiv.org/pdf/2404.10518
- ٷgithubhttps://github.com/tensorflow/modelsٷΪTensorFlow棬ṩΪPytorch棩
- Google
- ؼʣMobileNetV4, ͨģ, ƶ̬ϵͳ, 񾭼ܹ, , Mobile MQA, UIB, NAS
- ΢ŹںţAI https://github.com/AIFengheshu/Plug-play-modules
- Make sure that round down does not go down by more than 10%.
- starting depthwise conv
- expansion with 1x1 convs
- middle depthwise conv
- projection with 1x1 convs

## 2. 核心分析
### 类定义与参数
#### `class InvertedResidual`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, stride, expand_ratio, act, squeeze_exactation`

#### `class UniversalInvertedBottleneckBlock`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, out_channels, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample, stride, expand_ratio`

#### `class MultiQueryAttentionLayerWithDownSampling`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, dw_kernel_size, dropout`

#### `class MNV4layerScale`
- **描述**: 无文档说明。
- **初始化参数**: `init_value`

#### `class MultiHeadSelfAttentionBlock`
- **描述**: 无文档说明。
- **初始化参数**: `in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, use_layer_scale, use_multi_query, use_residual`

#### `class MobileNetV4`
- **描述**: 无文档说明。
- **初始化参数**: `model, num_classes`

## 3. 使用示例
```python
# 导入方式（参考）：from (2024) MobileNetV4 import ...

# 豸CPU  GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #  mobilenetv4_large ģ
    model = mobilenetv4_large(pretrained=False, num_classes=1000).to(device)
    # print(model)
    print("Model loaded successfully.")
    # ģݣbatch_size=1, 3ͨͼ񣬴СΪ224x224
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    # ģ
    with torch.no_grad():  # رݶȼ
        model.eval()  # ģΪģʽ
        output = model(input_tensor)
    # ӡ״
    print(f"Output shape: {output.shape}")
    # ӡģժҪѡ
    summary(model, input_size=(3, 224, 224))
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
