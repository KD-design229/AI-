# Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2007.02842](https://arxiv.org/pdf/2007.02842)
- **源文件**: `(AAAI 2019) AGCRN.py`

### 设计机制
- x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
- output shape [B, N, C]
- default cheb_k = 3
- x: B, num_nodes, input_dim
- state: B, num_nodes, hidden_dim
- shape of x: (B, T, N, D)
- shape of init_state: (num_layers, B, N, hidden_dim)
- current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
- output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
- last_state: (B, N, hidden_dim)

## 2. 核心分析
### 类定义与参数
#### `class AVWGCN`
- **描述**: 无文档说明。
- **初始化参数**: `dim_in, dim_out, cheb_k, embed_dim`

#### `class AGCRNCell`
- **描述**: 无文档说明。
- **初始化参数**: `node_num, dim_in, dim_out, cheb_k, embed_dim`

#### `class AVWDCRNN`
- **描述**: 无文档说明。
- **初始化参数**: `node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers`

#### `class AGCRN`
- **描述**: 无文档说明。
- **初始化参数**: `args`

## 3. 使用示例
```python
# 导入方式（参考）：from (AAAI 2019) AGCRN import ...

class Args:
        def __init__(self):
            self.num_nodes = 10  # 假设图中有10个节点
            self.input_dim = 1   # 每个节点的特征维度
            self.rnn_units = 64  # RNN单元的数量
            self.output_dim = 1  # 输出维度
            self.horizon = 3     # 预测未来3个时间步
            self.num_layers = 2  # 使用2层RNN
            self.cheb_k = 3      # 切比雪夫多项式的阶数
            self.embed_dim = 20  # 节点嵌入的维度

    # 实例化参数
    args = Args()

    # 实例化模型
    model = AGCRN(args)

    # 创建一个虚拟的输入数据
    input_tensor = torch.randn(1, 3, args.num_nodes, args.input_dim)
    print("Input tensor size: ", input_tensor.size())  # 打印输入尺寸

    # 创建虚拟的目标数据
    target_tensor = torch.randn(1, args.horizon, args.num_nodes, args.output_dim)
    print("Target tensor size:", target_tensor.size())  # 打印目标尺寸

    # 将模型转换为训练模式并进行前向传播
    model.train()
    output = model(input_tensor, target_tensor)
    print("Output size:       ", output.size())  # 打印输出尺寸
```

## 4. 适用任务
- **目标检测**
- **图像分类**
- **图像去噪**
- **时间序列预测**
- **目标跟踪**
- **去雨**
- **去雾**
- **去模糊**
- **图像融合**
- **语义分割/实例分割**
- **超分辨率**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
