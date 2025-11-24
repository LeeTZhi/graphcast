# Apple Silicon Mac (MPS) 训练指南

本指南说明如何在 Apple Silicon Mac (M1/M2/M3) 上使用 GPU 加速训练 ConvLSTM 模型。

## 什么是 MPS？

MPS (Metal Performance Shaders) 是 PyTorch 在 Mac 上使用 GPU 加速的方式。它利用 Apple Silicon 芯片的 GPU 来加速深度学习训练和推理。

## 系统要求

- **硬件**: Apple Silicon Mac (M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M2 Ultra, M3, M3 Pro, M3 Max)
- **操作系统**: macOS 12.3 或更高版本
- **PyTorch**: 1.12.0 或更高版本

## 检查 MPS 是否可用

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

如果输出 `MPS available: True`，说明您的系统支持 MPS。

## 使用 MPS 训练

### 方法 1: 自动检测（推荐）

使用 `--device auto`，脚本会自动选择最佳设备（优先级：cuda > mps > cpu）：

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --device auto
```

在 Mac 上，这会自动使用 MPS。

### 方法 2: 显式指定 MPS

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --device mps
```

## 完整训练示例

### 基础训练（仅下游区域）

```bash
python train_convlstm.py \
    --data ../MultiGridWF/MGWF/output/all_data_cleaned.nc \
    --output-dir checkpoints/baseline_mps \
    --device mps \
    --batch-size 4 \
    --num-epochs 100 \
    --use-amp
```

### 包含上游区域的训练

```bash
python train_convlstm.py \
    --data ../MultiGridWF/MGWF/output/all_data_cleaned.nc \
    --output-dir checkpoints/with_upstream_mps \
    --include-upstream \
    --device mps \
    --batch-size 4 \
    --num-epochs 100 \
    --use-amp
```

## 性能优化建议

### 1. 批量大小 (Batch Size)

MPS 的内存管理与 CUDA 不同。建议从较小的批量开始：

```bash
# 从 batch-size 4 开始
--batch-size 4

# 如果内存充足，可以增加到 8
--batch-size 8
```

### 2. 混合精度训练

MPS 支持混合精度训练，可以提高速度并减少内存使用：

```bash
--use-amp  # 启用混合精度（默认）
```

### 3. 数据加载器工作进程

在 Mac 上，建议使用较少的工作进程：

```bash
--num-workers 2  # 默认值，适合大多数 Mac
```

### 4. 梯度累积

如果遇到内存不足，使用梯度累积来模拟更大的批量：

```bash
--batch-size 2 \
--gradient-accumulation-steps 4  # 等效于 batch-size 8
```

## 性能对比

在 Apple Silicon Mac 上，MPS 相比 CPU 通常能提供 **2-5倍** 的训练速度提升：

| 设备 | 每个 Epoch 时间 | 相对速度 |
|------|----------------|----------|
| CPU  | ~150 秒        | 1.0x     |
| MPS  | ~30-75 秒      | 2-5x     |

实际性能取决于：
- Mac 型号（M1/M2/M3, Pro/Max/Ultra）
- 模型大小
- 批量大小
- 数据复杂度

## 常见问题

### Q: 为什么训练速度没有预期的快？

**A**: 可能的原因：
1. 批量太小 - 尝试增加 `--batch-size`
2. 数据加载瓶颈 - 确保数据在 SSD 上
3. 模型太小 - 小模型在 GPU 上的优势不明显

### Q: 遇到内存不足错误怎么办？

**A**: 尝试以下方法：
1. 减小批量大小：`--batch-size 2`
2. 使用梯度累积：`--gradient-accumulation-steps 4`
3. 减小模型大小：`--hidden-channels 16 32`
4. 关闭其他占用内存的应用

### Q: MPS 和 CUDA 有什么区别？

**A**: 
- **MPS**: Apple Silicon Mac 的 GPU 加速
- **CUDA**: NVIDIA GPU 的加速技术
- 两者不兼容，但 PyTorch 提供统一的 API

### Q: 可以在训练时切换设备吗？

**A**: 不可以。设备必须在训练开始前指定，训练过程中不能更改。

### Q: 如何监控 GPU 使用情况？

**A**: 使用 Activity Monitor（活动监视器）：
1. 打开 Activity Monitor
2. 选择 "Window" > "GPU History"
3. 观察 GPU 使用率

或使用命令行：
```bash
sudo powermetrics --samplers gpu_power -i 1000
```

## 故障排除

### 错误: "MPS backend out of memory"

**解决方案**:
```bash
# 减小批量大小
python train_convlstm.py ... --batch-size 2

# 或使用梯度累积
python train_convlstm.py ... --batch-size 2 --gradient-accumulation-steps 4
```

### 错误: "MPS is not available"

**解决方案**:
1. 确认您使用的是 Apple Silicon Mac
2. 更新 macOS 到 12.3 或更高版本
3. 更新 PyTorch：`pip install --upgrade torch`

### 训练速度慢

**解决方案**:
1. 确保使用 MPS：检查日志中是否显示 "Device: mps"
2. 增加批量大小（如果内存允许）
3. 确保数据在本地 SSD 上，而非网络驱动器
4. 关闭其他占用 GPU 的应用

## 最佳实践

1. **首次训练**: 使用较小的配置测试
   ```bash
   python train_convlstm.py \
       --data data.nc \
       --output-dir test \
       --device mps \
       --batch-size 2 \
       --num-epochs 2
   ```

2. **正式训练**: 根据测试结果调整参数
   ```bash
   python train_convlstm.py \
       --data data.nc \
       --output-dir production \
       --device mps \
       --batch-size 4 \
       --num-epochs 100 \
       --use-amp
   ```

3. **监控训练**: 观察日志中的损失值和训练时间

4. **保存检查点**: 脚本会自动保存最佳模型和定期检查点

## 参考资源

- [PyTorch MPS 官方文档](https://pytorch.org/docs/stable/notes/mps.html)
- [Apple Silicon 性能优化指南](https://developer.apple.com/metal/pytorch/)

## 总结

在 Apple Silicon Mac 上使用 MPS 训练 ConvLSTM 模型：

✅ **优点**:
- 比 CPU 快 2-5 倍
- 无需额外硬件
- 能耗更低

⚠️ **注意**:
- 内存管理与 CUDA 不同
- 某些操作可能不如 CUDA 优化
- 需要 macOS 12.3+

🚀 **开始训练**:
```bash
python train_convlstm.py \
    --data your_data.nc \
    --output-dir checkpoints/mps \
    --device mps
```
