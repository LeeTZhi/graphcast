# ConvLSTM 训练快速开始 (Mac 用户)

本指南帮助 Mac 用户快速开始使用 MPS (Apple Silicon GPU) 训练 ConvLSTM 模型。

## 前提条件

✅ Apple Silicon Mac (M1/M2/M3)
✅ macOS 12.3 或更高版本
✅ 已安装 PyTorch 1.12.0+

## 第一步：准备数据

如果您的数据有质量问题（NaN 值、异常值），先清理数据：

```bash
python scripts/clean_data_aggressive.py \
    --input 原始数据.nc \
    --output 清理后数据.nc
```

## 第二步：开始训练

### 基础训练（推荐新手）

```bash
python train_convlstm.py \
    --data 清理后数据.nc \
    --output-dir checkpoints/我的第一个模型 \
    --device mps \
    --batch-size 4 \
    --num-epochs 10
```

### 完整训练

```bash
python train_convlstm.py \
    --data 清理后数据.nc \
    --output-dir checkpoints/完整训练 \
    --device mps \
    --batch-size 4 \
    --num-epochs 100 \
    --use-amp
```

### 包含上游区域的训练

```bash
python train_convlstm.py \
    --data 清理后数据.nc \
    --output-dir checkpoints/包含上游 \
    --include-upstream \
    --device mps \
    --batch-size 4 \
    --num-epochs 100
```

## 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--data` | 数据文件路径 | 必填 |
| `--output-dir` | 输出目录 | 必填 |
| `--device` | 设备选择 | `mps` (Mac GPU) |
| `--batch-size` | 批量大小 | 4-8 |
| `--num-epochs` | 训练轮数 | 100 |
| `--learning-rate` | 学习率 | 0.001 |
| `--use-amp` | 混合精度训练 | 推荐启用 |
| `--include-upstream` | 包含上游区域 | 对比实验用 |

## 监控训练

训练日志会保存在输出目录的 `training.log` 文件中。

查看实时日志：
```bash
tail -f checkpoints/我的第一个模型/training.log
```

## 训练输出

训练完成后，输出目录包含：

- `best_model.pt` - 最佳模型（验证损失最低）
- `checkpoint_epoch_N.pt` - 定期检查点
- `normalizer.pkl` - 数据归一化参数
- `training.log` - 训练日志

## 性能优化

### 如果训练太慢

1. 确认使用了 MPS：
   ```bash
   # 日志中应该显示 "Device: mps"
   grep "Device:" checkpoints/*/training.log
   ```

2. 增加批量大小（如果内存允许）：
   ```bash
   --batch-size 8
   ```

### 如果内存不足

1. 减小批量大小：
   ```bash
   --batch-size 2
   ```

2. 使用梯度累积：
   ```bash
   --batch-size 2 --gradient-accumulation-steps 4
   ```

3. 减小模型大小：
   ```bash
   --hidden-channels 16 32
   ```

## 常见问题

### Q: 如何知道训练是否正常？

A: 查看日志中的损失值（loss）应该逐渐下降：
```
Epoch 1/100: train_loss=1.2339, val_loss=1.1394
Epoch 2/100: train_loss=1.1349, val_loss=1.1035  ← 损失在下降，正常！
```

### Q: 训练需要多长时间？

A: 在 M1/M2 Mac 上：
- 每个 epoch: 约 30-75 秒
- 100 epochs: 约 1-2 小时

### Q: 可以中断训练吗？

A: 可以！按 `Ctrl+C` 中断，脚本会自动保存检查点。

恢复训练：
```bash
python train_convlstm.py \
    --data 数据.nc \
    --output-dir checkpoints/我的模型 \
    --resume checkpoints/我的模型/interrupted_checkpoint.pt
```

## 下一步

训练完成后，可以：

1. **评估模型**: 查看 `EVALUATION_GUIDE.md`
2. **进行预测**: 查看 `INFERENCE_GUIDE.md`
3. **可视化结果**: 使用 `visualization.py`

## 获取帮助

查看完整文档：
- 训练指南: `TRAINING_GUIDE.md`
- MPS 详细说明: `MPS_GUIDE.md`
- 主文档: `README.md`

查看所有参数：
```bash
python train_convlstm.py --help
```

## 示例：完整工作流程

```bash
# 1. 清理数据
python scripts/clean_data_aggressive.py \
    --input ../MultiGridWF/MGWF/output/all_data.nc \
    --output ../MultiGridWF/MGWF/output/all_data_cleaned.nc

# 2. 训练基线模型
python train_convlstm.py \
    --data ../MultiGridWF/MGWF/output/all_data_cleaned.nc \
    --output-dir checkpoints/baseline \
    --device mps \
    --batch-size 4 \
    --num-epochs 100

# 3. 训练包含上游的模型
python train_convlstm.py \
    --data ../MultiGridWF/MGWF/output/all_data_cleaned.nc \
    --output-dir checkpoints/with_upstream \
    --include-upstream \
    --device mps \
    --batch-size 4 \
    --num-epochs 100

# 4. 比较两个模型的性能
# （使用 evaluation.py，详见 EVALUATION_GUIDE.md）
```

祝训练顺利！🚀
