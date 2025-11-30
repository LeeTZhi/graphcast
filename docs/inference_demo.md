# 1. 环境准备

训练和推理需要依赖Pytorch以及xarray等python包，所以需要预先准备，可以使用uv进行安装，基础依赖的包参见 `requirements_pytorch.txt` 文件。建议创建一个虚拟环境进行安装，安装参考脚本如下：

```bash
# 创建虚拟环境
uv venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
uv pip install -r requirements_pytorch.txt
```

# 2. 数据预处理

数据已经被整体加载为xarray，保存nc文件（网盘里面的2025.11_data下）。由于内部存在大量的NaN和超范围的异常数据，所以需要数据清洗，清洗使用脚本 scripts/clean_data_aggressive.py。 具体用法
```python
python3 scripts/clean_data_aggressive.py --input ../Data/2019-2024_all.nc \
    --output ../Data/2019-2024_all_clean.nc \
        --create-mask
```

将清洗后的数据保存为2019-2024_all_clean.nc，并且针对NaN数据创建了Mask，后面使用时候会屏蔽掉。

# 3. 模型推理

模型文件保存在网盘 `models` 目录下。推理脚本 `run_inference_convlstm.py` 支持多种模型架构和配置选项。

## 3.1 支持的模型类型

脚本支持自动检测以下模型架构：
- `shallow`: 基础 ConvLSTM-UNet（2层）
- `deep`: 深度 ConvLSTM-UNet（4层，带正则化）
- `dual_stream`: 双流 ConvLSTM-UNet（上下游双区域，2层）
- `dual_stream_deep`: 深度双流 ConvLSTM-UNet（上下游双区域，4层）

模型类型会从 checkpoint 自动检测，也可以通过 `--model-type` 参数手动指定（适用于旧版本 checkpoint）。

## 3.2 基础推理示例

### 单区域模型推理

```bash
# 基础推理 - 生成预测结果
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/baseline

# 带可视化的推理
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/baseline \
    --visualize \
    --viz-timesteps 0 5 10
```

### 双流模型推理（上下游双区域）

```bash
# 双流模型会自动检测并启用上游区域
python3 run_inference_convlstm.py \
    --checkpoint ./models/dual_stream_model.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/dual_stream \
    --visualize

# 也可以手动指定（如果自动检测失败）
python3 run_inference_convlstm.py \
    --checkpoint ./models/dual_stream_model.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/dual_stream \
    --include-upstream \
    --model-type dual_stream_deep \
    --visualize
```

## 3.3 时间过滤选项

### 指定时间范围

```bash
# 预测指定时间段
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/time_range \
    --start-time 2021-06-01T00:00:00 \
    --end-time 2021-08-31T23:00:00 \
    --visualize
```

### 指定特定时刻

```bash
# 预测特定时刻（自动加载历史数据）
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/specific_times \
    --specific-times 2021-08-11T02:00:00 2021-08-15T14:00:00 \
    --visualize

# 注意：使用 --specific-times 时，脚本会自动加载所需的历史数据
# 例如：window_size=6 时，预测 2021-08-11T02:00:00 会自动加载
# 从 2021-08-10T20:00:00 到 2021-08-11T02:00:00 的数据
```

## 3.4 模型对比实验

```bash
# 对比两个模型的预测效果
python3 run_inference_convlstm.py \
    --checkpoint ./models/baseline_model.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/comparison \
    --compare-checkpoint ./models/dual_stream_model.pt \
    --compare-normalizer ./models/normalizer.pkl \
    --compare-upstream \
    --exp1-name "单区域模型" \
    --exp2-name "双流模型" \
    --visualize
```

## 3.5 可视化选项

```bash
# 自定义可视化参数
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/custom_viz \
    --visualize \
    --viz-timesteps 0 3 6 9 12 \
    --viz-format png \
    --viz-dpi 300 \
    --viz-vmin 0.0 \
    --viz-vmax 500.0
```

可视化选项说明：
- `--viz-timesteps`: 指定要可视化的时间步索引（默认：前5个）
- `--viz-format`: 图片格式（png/pdf/jpg/svg，默认：png）
- `--viz-dpi`: 图片分辨率（默认：150）
- `--viz-vmin/--viz-vmax`: 降水量色标范围（默认：0-500 mm）

## 3.6 区域配置

```bash
# 自定义下游和上游区域范围
python3 run_inference_convlstm.py \
    --checkpoint ./models/dual_stream_model.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/custom_region \
    --downstream-lat-min 25.0 \
    --downstream-lat-max 40.0 \
    --downstream-lon-min 110.0 \
    --downstream-lon-max 125.0 \
    --upstream-lat-min 25.0 \
    --upstream-lat-max 50.0 \
    --upstream-lon-min 70.0 \
    --upstream-lon-max 110.0 \
    --visualize
```

## 3.7 设备选择

```bash
# 自动选择最佳设备（默认）
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/auto_device \
    --device auto

# 使用 Apple Silicon GPU (MPS)
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/mps \
    --device mps

# 使用 CUDA GPU
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/cuda \
    --device cuda

# 使用 CPU
python3 run_inference_convlstm.py \
    --checkpoint ./models/single_region_deep_1130.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/cpu \
    --device cpu
```

## 3.8 输出结果

推理完成后，结果保存在指定的输出目录：

```
outputs/
├── inference.log              # 推理日志
├── predictions.nc             # 预测结果（NetCDF格式）
└── visualizations/            # 可视化图片
    ├── Experiment_1_timestep_0_truth_vs_pred.png
    ├── Experiment_1_timestep_0_error.png
    ├── comparison_timestep_0_comparison.png
    └── ...
```

可视化图片类型：
- `truth_vs_pred`: 真实值与预测值对比
- `error`: 预测误差分布图
- `comparison`: 多模型对比（需要 `--compare-checkpoint`）
- `multi_error`: 多模型误差对比

## 3.9 完整参数说明

查看所有可用参数：

```bash
python3 run_inference_convlstm.py --help
```

主要参数分类：
- **必需参数**: `--checkpoint`, `--normalizer`, `--data`, `--output-dir`
- **区域配置**: `--include-upstream`, `--downstream-*`, `--upstream-*`
- **推理配置**: `--model-type`, `--window-size`, `--target-offset`, `--batch-size`
- **时间过滤**: `--start-time`, `--end-time`, `--specific-times`
- **模型对比**: `--compare-checkpoint`, `--compare-normalizer`, `--compare-upstream`
- **可视化**: `--visualize`, `--viz-timesteps`, `--viz-format`, `--viz-dpi`
- **输出选项**: `--save-predictions`, `--predictions-filename`
- **设备和日志**: `--device`, `--log-level`

## 3.10 常见问题

### Q1: 如何处理日期格式错误？

确保日期格式正确，注意月份天数：
```bash
# 错误：6月只有30天
--specific-times 2020-06-31T00:00:00  # ❌

# 正确
--specific-times 2020-06-30T00:00:00  # ✓
```

### Q2: 双流模型是否需要手动指定 `--include-upstream`？

不需要。脚本会自动检测模型类型，如果是双流模型会自动启用上游区域。

### Q3: 如何加载旧版本 checkpoint？

使用 `--model-type` 手动指定模型类型：
```bash
python3 run_inference_convlstm.py \
    --checkpoint ./models/old_model.pt \
    --normalizer ./models/normalizer.pkl \
    --data ../../Data/2019-2023_all_clean.nc \
    --output-dir ./outputs/old_model \
    --model-type dual_stream_deep
```

### Q4: 如何预测特定时刻但不知道需要多少历史数据？

使用 `--specific-times` 时，脚本会根据 `--window-size` 自动加载所需的历史数据，无需手动计算。
