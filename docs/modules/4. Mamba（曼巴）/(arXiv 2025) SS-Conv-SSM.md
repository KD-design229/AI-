# (arXiv 2025) SS-Conv-SSM

## 1. 模块简介
- **源文件**: `(arXiv 2025) SS-Conv-SSM.py`

### 设计机制
- an alternative for mamba_ssm (in which causal_conv1d is needed)
- d_state="auto", # 20240109
- self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
- self.selective_scan = selective_scan_fn
- Initialize special dt projection to preserve variance at initialization

## 2. 核心分析
### 类定义与参数
#### `class SS2D`
- **描述**: 无文档说明。
- **初始化参数**: `d_model, d_state, d_conv, expand, dt_rank, dt_min, dt_max, dt_init, dt_scale, dt_init_floor, dropout, conv_bias, bias, device, dtype`

#### `class SS_Conv_SSM`
- **描述**: 无文档说明。
- **初始化参数**: `hidden_dim, drop_path, norm_layer, attn_drop_rate, d_state`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2025) SS-Conv-SSM import ...

# 设置测试参数
    B = 1               # 批次大小
    C = 32              # 通道数
    H = 224             # 高度
    W = 224             # 宽度
    d_state = 16        # 状态维度
    drop_path = 0.1     # Drop路径的概率
    attn_drop_rate = 0  # 注意力丢弃率

    # 创建一个随机输入张量，形状为 (B, H, W, C)
    x = torch.randn(B, H, W, C).cuda()
    # 初始化
    model = SS_Conv_SSM(hidden_dim=C, d_state=d_state, drop_path=drop_path, attn_drop_rate=attn_drop_rate).cuda()
    print(model)
    # 运行模型前向传播
    output = model(x)
    print("\n微信公众号: AI缝合术!\n")
    # 打印输出的形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
