# (arXiv 2025) FFTTransformerEncoderBlock

## 1. 模块简介
- **源文件**: `(arXiv 2025) FFTTransformerEncoderBlock.py`

### 设计机制
- Generate binary tensor mask; shape: (batch_size, 1, 1, ..., 1)
- 频域的 FFT 频率桶数量: (seq_len//2 + 1)
- 基础乘法滤波器: 每个注意力头和频率桶一个
- 基础加性偏置: 作为频率幅度的学习偏移
- 自适应 MLP: 每个头部和频率桶生成 2 个值(缩放因子和偏置)
- 预归一化层,提高傅里叶变换的稳定性
- 对幅度进行非线性变换,GELU 提供平滑的非线性
- 计算缩放因子,防止除零错误
- 预归一化,提高频域变换的稳定性
- 重新排列张量以分离不同的注意力头,形状变为 (B, num_heads, seq_len, head_dim)
- 沿着序列维度计算 FFT,结果为复数张量,形状为 (B, num_heads, freq_bins, head_dim)

## 2. 核心分析
### 类定义与参数
#### `class DropPath`
- **描述**: DropPath module that performs stochastic depth.
- **初始化参数**: `drop_prob`

#### `class MultiHeadSpectralAttention`
- **描述**: 无文档说明。
- **初始化参数**: `embed_dim, seq_len, num_heads, dropout, adaptive`

#### `class FFTTransformerEncoderBlock`
- **描述**: 无文档说明。
- **初始化参数**: `embed_dim, mlp_ratio, dropout, attention_module, drop_path`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2025) FFTTransformerEncoderBlock import ...

# 参数设置
    batch_size = 1      # 批大小
    seq_len = 224 * 224 # 序列长度
    embed_dim = 32      # 嵌入维度
    num_heads = 4       # 注意力头数

    # 创建随机输入张量 (batch_size, seq_len, embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 初始化 MultiHeadSpectralAttention
    attention_module = MultiHeadSpectralAttention(embed_dim=embed_dim, seq_len=seq_len, num_heads=num_heads)

    # 初始化 TransformerEncoderBlock
    transformer_block = FFTTransformerEncoderBlock(embed_dim=embed_dim, attention_module=attention_module)
    print(transformer_block)
    print("微信公众号: AI缝合术!")

    # 前向传播测试
    output = transformer_block(x)

    # 打印输出形状
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
```

## 4. 适用任务
- **Transformer相关任务**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
