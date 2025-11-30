# 双流ConvLSTM架构说明

## 问题背景

在使用多个区域（上游和下游）进行训练时，如果两个区域的空间维度不同（例如上游区域101×61，下游区域151×81），直接在空间维度上拼接会导致维度不匹配错误：

```
ValueError: all the input array dimensions except for the concatenation axis must match exactly, 
but along dimension 2, the array at index 0 has size 101 and the array at index 1 has size 61
```

## 解决方案：双流架构

新的双流架构通过以下方式解决这个问题：

### 1. 架构设计

```
上游输入 [B,T,C,H_up,W_up]     下游输入 [B,T,C,H_down,W_down]
         |                              |
    上游编码器                      下游编码器
         |                              |
    下采样 (Pool)                  下采样 (Pool)
         |                              |
    [B,C,H_up/2,W_up/2]           [B,C,H_down/2,W_down/2]
         |                              |
         +--------- 调整尺寸 -----------+
                        |
                  拼接 (Concat)
                        |
                   瓶颈层融合
                        |
                  自注意力机制
                        |
                    上采样
                        |
                  跳跃连接 (下游)
                        |
                    解码器
                        |
              输出 [B,1,H_down,W_down]
```

### 2. 关键特性

- **独立编码**：上游和下游区域通过独立的编码器处理，可以处理不同的空间维度
- **瓶颈融合**：在瓶颈层通过插值调整尺寸后拼接特征，实现信息融合
- **下游重建**：解码器只重建下游区域，使用下游编码器的跳跃连接
- **灵活性**：支持单流模式（仅下游）和双流模式（上游+下游）

### 3. 模型类型

#### DualStreamConvLSTMUNet (浅层双流)
- 2层编码器 + 瓶颈层 + 1层解码器
- 默认隐藏通道：[64, 128]
- 参数量：约320万

#### DeepDualStreamConvLSTMUNet (深层双流)
- 3层编码器 + 瓶颈层 + 3层解码器
- 默认隐藏通道：[64, 128, 256, 512]
- 参数量：约5500万
- 支持Dropout正则化

## 使用方法

### 1. 数据加载

数据集会自动返回正确的格式：

```python
# 单流模式 (include_upstream=False)
dataset = ConvLSTMDataset(
    data=data,
    window_size=6,
    region_config=region_config,
    include_upstream=False
)
# 返回: (input_tensor, target_tensor)
# input_tensor: [T, C, H, W]

# 双流模式 (include_upstream=True)
dataset = ConvLSTMDataset(
    data=data,
    window_size=6,
    region_config=region_config,
    include_upstream=True
)
# 返回: (input_dict, target_tensor)
# input_dict: {'downstream': [T,C,H_d,W_d], 'upstream': [T,C,H_u,W_u]}
```

### 2. 模型创建

```python
from convlstm.model_dual_stream import DualStreamConvLSTMUNet
from convlstm.model_dual_stream_deep import DeepDualStreamConvLSTMUNet

# 浅层双流模型
model = DualStreamConvLSTMUNet(
    input_channels=56,
    hidden_channels=[64, 128],
    output_channels=1,
    use_attention=True,
    use_group_norm=True,
    dropout_rate=0.0
)

# 深层双流模型
model = DeepDualStreamConvLSTMUNet(
    input_channels=56,
    hidden_channels=[64, 128, 256, 512],
    output_channels=1,
    use_attention=True,
    use_group_norm=True,
    dropout_rate=0.2
)
```

### 3. 训练命令

```bash
# 使用双流模型训练
python train_convlstm.py \
    --data /path/to/data.nc \
    --output-dir ./output \
    --model-type dual_stream \
    --include-upstream \
    --hidden-channels 64 128 \
    --batch-size 4 \
    --num-epochs 100

# 使用深层双流模型
python train_convlstm.py \
    --data /path/to/data.nc \
    --output-dir ./output \
    --model-type dual_stream_deep \
    --include-upstream \
    --hidden-channels 64 128 256 512 \
    --dropout-rate 0.2 \
    --batch-size 2 \
    --num-epochs 100
```

## 技术细节

### 1. 特征融合策略

在瓶颈层，上游特征通过双线性插值调整到与下游特征相同的空间尺寸：

```python
# 上游特征: [B, C, H_up//2, W_up//2]
# 下游特征: [B, C, H_down//2, W_down//2]

up_resized = F.interpolate(
    up_encoded,
    size=(down_encoded.size(2), down_encoded.size(3)),
    mode='bilinear',
    align_corners=True
)

# 拼接: [B, 2*C, H_down//2, W_down//2]
fused = torch.cat([down_encoded, up_resized], dim=1)
```

### 2. 跳跃连接

只使用下游编码器的特征进行跳跃连接，确保解码器重建的是下游区域：

```python
# 解码器输入 = 上采样的瓶颈特征 + 下游编码器特征
decoder_input = torch.cat([upsampled_bottleneck, downstream_encoder_output], dim=1)
```

### 3. 自适应模式

模型自动检测输入类型：

```python
def forward(self, x):
    if isinstance(x, dict):
        # 双流模式
        return self._forward_dual_stream(x)
    else:
        # 单流模式
        return self._forward_single_stream(x)
```

## 优势

1. **灵活性**：支持不同空间维度的区域
2. **可扩展性**：可以轻松添加更多区域
3. **效率**：只在瓶颈层融合，减少计算量
4. **兼容性**：向后兼容单流模式

## 注意事项

1. 双流模式需要更多GPU内存（约1.5-2倍）
2. 建议使用较小的batch_size（2-4）
3. 深层模型需要更多训练时间
4. 可以通过dropout_rate控制正则化强度
