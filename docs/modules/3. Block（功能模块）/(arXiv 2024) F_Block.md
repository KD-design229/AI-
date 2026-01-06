# ATFNet: Adaptive Time-Frequency Ensembled Network for Long-term Time Series Forecasting

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2404.05192](https://arxiv.org/pdf/2404.05192)
- **源文件**: `(arXiv 2024) F_Block.py`

## 2. 核心分析
### 类定义与参数
#### `class ComplexLN`
- **描述**: 无文档说明。
- **初始化参数**: `C`

#### `class ComplexLinear`
- **描述**: 无文档说明。
- **初始化参数**: `in_features, out_features`

#### `class ComplexAttention`
- **描述**: 无文档说明。

#### `class ComplexAttentionLayer`
- **描述**: 无文档说明。
- **初始化参数**: `d_model, n_heads, attention, d_keys, d_values`

#### `class ComplexEncoderLayer`
- **描述**: 无文档说明。
- **初始化参数**: `attention, d_model, d_ff, dropout`

#### `class ComplexEncoder`
- **描述**: 无文档说明。
- **初始化参数**: `attn_layers`

#### `class CompEncoderBlock`
- **描述**: 无文档说明。
- **初始化参数**: `configs, extended`

#### `class F_Block`
- **描述**: 无文档说明。
- **初始化参数**: `configs`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2024) F_Block import ...

# 定义一个示例配置对象
    class Configs:
        def __init__(self):
            self.seq_len = 96  # 输入序列长度
            self.pred_len = 96  # 预测序列长度
            self.d_model = 512  # 模型的维度
            self.factor = 5  # 用于缩放注意力机制的因子
            self.n_heads = 8  # 注意力头的数量
            self.e_layers = 3  # 编码器的层数
            self.d_ff = 2048  # 前馈神经网络的维度
            self.dropout = 0.1  # dropout的概率
            self.activation = 'gelu'  # 激活函数
            self.enc_in = 7  # 编码器输入的特征数量
            self.dec_in = 7  # 解码器输入的特征数量
            self.c_out = 1  # 输出的特征数量
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU还是CPU
            self.fnet_d_ff = 1024  # 频域前馈神经网络的维度
            self.fnet_d_model = 512  # 频域模型的维度
            self.complex_dropout = 0.1  # 复数dropout的概率
            self.fnet_layers = 2  # 频域网络的层数
            self.is_emb = False  # 是否使用嵌入层

    configs = Configs()

    block = F_Block(configs).to(configs.device)  # 初始化并将模型移动到指定设备

    x_enc = torch.rand(2, configs.seq_len, configs.enc_in).to(configs.device)  # (batch_size, seq_len, n_vars)


    output = block(x_enc)

    # 打印输入和输出张量的尺寸
    print("x_enc size:     ", x_enc.size())

    print("Output size:    ", output.size())
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
