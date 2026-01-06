# MoH: Multi-Head Attention as Mixture-of-Head Attention

## 1. 模块简介
- **论文地址**: [https://arxiv.org/pdf/2410.11842?](https://arxiv.org/pdf/2410.11842?)
- **源文件**: `(arXiv 2024) MoHAttention.py`

### 设计机制
- 中文题目:  MoH：多头注意力作为混合头注意力
- 官方github：https://github.com/SkyworkAI/MoH
- 所属机构：北京大学深圳电子与计算机工程学院，深圳鹏城实验室，深圳兔智科技，新加坡昆仑2050研究与Skywork AI
- 关键词：多头注意力机制、混合注意力机制、图像分类、图像生成、大语言模型
- assert dim % num_heads == 0, 'dim should be divisible by num_heads'

## 2. 核心分析
### 类定义与参数
#### `class MoHAttention`
- **描述**: 无文档说明。
- **初始化参数**: `dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer, shared_head, routed_head, head_dim`

## 3. 使用示例
```python
# 导入方式（参考）：from (arXiv 2024) MoHAttention import ...

main()
```

## 4. 适用任务
- **图像分类**
- **注意力机制应用**
- **集成推荐**: 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
