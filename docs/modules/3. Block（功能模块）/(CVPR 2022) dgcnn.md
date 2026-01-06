# (CVPR 2022) dgcnn

## 1. 模块简介
- **源文件**: `(CVPR 2022) dgcnn.py`

### 设计机制
- x = x.squeeze()
- (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
- Test the code.

## 2. 核心分析
### 类定义与参数
#### `class DGCNN`
- **描述**: 无文档说明。
- **初始化参数**: `emb_dims, input_shape`

## 3. 使用示例
```python
# 导入方式（参考）：from (CVPR 2022) dgcnn import ...

# Test the code.
    x = torch.rand((10, 1024, 3)).cuda()

    dgcnn = DGCNN().cuda()
    y = dgcnn(x)
    print("\nInput Shape of DGCNN: ", x.shape, "\nOutput Shape of DGCNN: ", y.shape)
```

## 4. 适用场景
- 该模块适用于各类计算机视觉任务，如图像分类、目标检测和语义分割等。
- 特别推荐在需要增强模型对特定特征（如空间位置、通道相关性或多尺度信息）的敏感度时使用。
