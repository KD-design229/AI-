# Rewrite the Stars

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2403.19967](https://arxiv.org/pdf/2403.19967)
- **源文件**: `(CVPR 2024) StarNet.py`

### 设计机制
- 中文题目:  重写星操作
- 官方github：https://github.com/ma-xu/Rewrite-the-Stars
- 所属机构：东北大学，微软
- 关键词：星操作、网络设计、StarNet、高效网络、核技巧
- stem layer
- build stages

## 2. 核心分析
### 类定义与参数
#### `class ConvBN`
- **描述**: 无文档说明。
- **初始化参数**: `in_planes, out_planes, kernel_size, stride, padding, dilation, groups, with_bn`

#### `class Block`
- **描述**: 无文档说明。
- **初始化参数**: `dim, mlp_ratio, drop_path`

#### `class StarNet`
- **描述**: 无文档说明。
- **初始化参数**: `base_dim, depths, mlp_ratio, drop_path_rate, num_classes`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2024) StarNet import ...

# 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载 starnet_s1 模型
    model = starnet_s1(pretrained=False, num_classes=1000).to(device)
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
- **通用骨干网络/特征提取**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
