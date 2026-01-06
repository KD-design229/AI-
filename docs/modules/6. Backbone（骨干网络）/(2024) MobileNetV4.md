# MobileNetV4: Universal Models for the Mobile Ecosystem

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2404.10518](https://arxiv.org/pdf/2404.10518)
- **源文件**: `(2024) MobileNetV4.py`

### 设计机制
- 中文题目: MobileNetV4：移动生态系统的通用模型
- 官方github：https://github.com/tensorflow/models（官方代码为TensorFlow版，本文提供的为Pytorch版）
- 所属机构：Google
- 关键词：MobileNetV4, 通用模型, 移动生态系统, 神经架构搜索, 帕累托最优, Mobile MQA, UIB, NAS
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

# 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载 mobilenetv4_large 模型
    model = mobilenetv4_large(pretrained=False, num_classes=1000).to(device)
    # print(model)
    print("Model loaded successfully.")
    # 模拟输入数据（batch_size=1, 3通道图像，大小为224x224）
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    # 模型推理
    with torch.no_grad():  # 关闭梯度计算
        model.eval()  # 设置模型为评估模式
        output = model(input_tensor)
    # 打印输出结果形状
    print(f"Output shape: {output.shape}")
    # 打印模型摘要（可选）
    summary(model, input_size=(3, 224, 224))
```

## 4. 适用任务
- **注意力机制应用**
- **通用骨干网络/特征提取**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
