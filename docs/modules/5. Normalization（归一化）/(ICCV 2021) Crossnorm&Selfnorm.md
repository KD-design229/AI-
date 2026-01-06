# CrossNorm and SelfNorm for Generalization under Distribution Shifts

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2102.02811](https://arxiv.org/pdf/2102.02811)
- **源文件**: `(ICCV 2021) Crossnorm&Selfnorm.py`

### 设计机制
- 中文题目：CrossNorm和SelfNorm在分布偏移下的泛化
- 官方github：https://github.com/amazon-science/crossnorm-selfnorm
- 所属机构：亚马逊网络服务，罗格斯大学
- eps is a small value added to the variance to avoid divide-by-zero.

## 2. 核心分析
### 类定义与参数
#### `class CrossNorm`
- **描述**: CrossNorm module
- **初始化参数**: `crop, beta`

#### `class SelfNorm`
- **描述**: SelfNorm module
- **初始化参数**: `chan_num, is_two`

## 3. 使用示例
```python
# 导入方式（参考）：from (ICCV 2021) Crossnorm&Selfnorm import ...

# 生成随机张量，形状为 (batch_size, channels, height, width)
    # 注意batch_size太小无法计算均值和方差，建议2以上
    batch_size, channels, height, width = 2, 32, 256, 256
    x = torch.randn(batch_size, channels, height, width)
    print("input shape:", x.shape)

    # 测试 CrossNorm
    cross_norm = CrossNorm(crop='style', beta=1)
    cross_norm.train()  # 设置为训练模式
    cross_norm.active = True  # 激活 CrossNorm
    x_cn = cross_norm(x)  # 使用 CrossNorm 对输入张量进行处理
    print("CrossNorm output shape:", x_cn.shape)
   
    # 测试 SelfNorm
    self_norm = SelfNorm(chan_num=channels, is_two=True)
    self_norm.train()  # 设置为训练模式
    x_sn = self_norm(x)  # 使用 SelfNorm 对输入张量进行处理
    print("SelfNorm output shape:", x_sn.shape)
```

## 4. 适用任务
- **归一化/模型稳定化**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
