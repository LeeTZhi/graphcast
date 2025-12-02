# Multi-Variable Prediction Guide

This guide explains how to use the ConvLSTM weather prediction system in multi-variable mode, which predicts all atmospheric variables (DPT, GPH, TEM, U, V) plus precipitation simultaneously.

## Table of Contents

- [Overview](#overview)
- [Why Multi-Variable Prediction?](#why-multi-variable-prediction)
- [Quick Start](#quick-start)
- [Configuration Parameters](#configuration-parameters)
- [Training](#training)
- [Rolling Forecasts](#rolling-forecasts)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Loss Function](#loss-function)
- [Data Format](#data-format)
- [Backward Compatibility](#backward-compatibility)
- [Troubleshooting](#troubleshooting)

## Overview

Multi-variable mode extends the ConvLSTM model to predict all atmospheric variables simultaneously:
- **5 atmospheric variables**: DPT (dew point), GPH (geopotential height), TEM (temperature), U (eastward wind), V (northward wind)
- **11 pressure levels**: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 100 hPa
- **1 surface variable**: Precipitation
- **Total**: 56 output channels (5 × 11 + 1)

### Key Benefits

1. **Better Generalization**: Model learns underlying atmospheric dynamics, not just precipitation patterns
2. **Physical Consistency**: Predictions respect physical relationships between variables
3. **Rolling Forecasts**: Enables multi-step predictions by feeding outputs back as inputs
4. **Comprehensive Output**: Provides full atmospheric state, not just precipitation

## Why Multi-Variable Prediction?

Traditional single-variable models only predict precipitation, treating it as an isolated quantity. However, precipitation is the result of complex atmospheric processes involving temperature, humidity, wind, and pressure patterns.

Multi-variable prediction offers several advantages:

1. **Learning Atmospheric Dynamics**: By predicting all variables, the model learns the underlying physics that drives weather patterns
2. **Improved Accuracy**: Understanding atmospheric relationships leads to better precipitation forecasts
3. **Extended Predictions**: Rolling forecasts become possible by using predicted atmospheric states as inputs
4. **Richer Information**: Provides complete atmospheric state for downstream applications

## Quick Start

### Training a Multi-Variable Model

```bash
# Basic multi-variable training
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/multi_variable \
    --multi-variable \
    --batch-size 4 \
    --num-epochs 100
```

### Running Inference

```bash
# Single-step prediction
python run_inference_convlstm.py \
    --checkpoint checkpoints/multi_variable/best_model.pt \
    --data data/test_data.nc \
    --output-dir predictions/multi_variable \
    --visualize
```

### Rolling Forecast

```bash
# Multi-step rolling forecast (6 steps = 3 days)
python run_inference_convlstm.py \
    --checkpoint checkpoints/multi_variable/best_model.pt \
    --data data/test_data.nc \
    --output-dir predictions/rolling \
    --rolling-steps 6 \
    --visualize
```

## Configuration Parameters

### Core Parameters

#### `--multi-variable`
- **Type**: Flag (boolean)
- **Default**: False (single-variable mode)
- **Description**: Enable multi-variable prediction mode (56 output channels)
- **Example**: `--multi-variable`

#### `--precip-loss-weight`
- **Type**: Float
- **Default**: 10.0
- **Description**: Weight multiplier for precipitation loss vs atmospheric variables
- **Range**: 1.0 - 20.0 (typical)
- **Example**: `--precip-loss-weight 15.0`
- **Notes**: Higher values prioritize precipitation accuracy

#### `--max-rollout-steps`
- **Type**: Integer
- **Default**: 6
- **Description**: Maximum number of rolling forecast steps
- **Range**: 1 - 6 (6 steps = 3 days at 12-hour intervals)
- **Example**: `--max-rollout-steps 4`

#### `--enable-rollout-training`
- **Type**: Flag (boolean)
- **Default**: False
- **Description**: Enable autoregressive training for rolling forecasts
- **Example**: `--enable-rollout-training`
- **Notes**: Requires more memory; use with gradient accumulation

### Configuration File

You can also specify parameters in a YAML configuration file:

```yaml
# config/multi_variable.yaml
multi_variable: true
precip_loss_weight: 10.0
max_rollout_steps: 6
enable_rollout_training: false

# Standard parameters
batch_size: 4
learning_rate: 0.001
num_epochs: 100
hidden_channels: [32, 64]
```

Load with:
```bash
python train_convlstm.py --config config/multi_variable.yaml
```

## Training

### Basic Training

Train a multi-variable model with default settings:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/multi_variable_basic \
    --multi-variable \
    --batch-size 4 \
    --num-epochs 100 \
    --use-amp
```

### Training with Rolling Forecast Supervision

Enable autoregressive training to improve multi-step predictions:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/multi_variable_rolling \
    --multi-variable \
    --enable-rollout-training \
    --max-rollout-steps 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --num-epochs 100 \
    --use-amp
```

**Notes on Rolling Training:**
- Requires more memory due to autoregressive backpropagation
- Use smaller batch sizes and gradient accumulation
- Start with fewer rollout steps (2-3) before increasing
- Gradient clipping helps with stability

### Custom Loss Weighting

Adjust the balance between precipitation and atmospheric variables:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/multi_variable_weighted \
    --multi-variable \
    --precip-loss-weight 15.0 \
    --high-precip-threshold 15.0 \
    --high-precip-weight 5.0 \
    --batch-size 4 \
    --num-epochs 100 \
    --use-amp
```

### Memory-Optimized Training

For GPUs with limited memory (8-12GB):

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/multi_variable_memory_opt \
    --multi-variable \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --hidden-channels 32 64 \
    --use-amp \
    --num-workers 2 \
    --num-epochs 100
```

### Model Variants

All four model architectures support multi-variable mode:

```bash
# Base model (default)
python train_convlstm.py --multi-variable --model-type base

# Deep model (more layers)
python train_convlstm.py --multi-variable --model-type deep

# Dual-stream model (with upstream region)
python train_convlstm.py --multi-variable --model-type dual_stream --include-upstream

# Deep dual-stream model
python train_convlstm.py --multi-variable --model-type dual_stream_deep --include-upstream
```

## Rolling Forecasts

Rolling forecasts enable multi-step predictions by iteratively feeding predictions back as inputs.

### How Rolling Forecasts Work

1. **Initial Input**: Start with 6 timesteps of atmospheric data
2. **Predict**: Model predicts next timestep (all 56 channels)
3. **Update Window**: Remove oldest timestep, append prediction
4. **Repeat**: Continue for desired number of steps (up to 6)

```
Initial: [t-5, t-4, t-3, t-2, t-1, t0] → Predict t1
Step 1:  [t-4, t-3, t-2, t-1, t0, t1] → Predict t2
Step 2:  [t-3, t-2, t-1, t0, t1, t2] → Predict t3
...
```

### Basic Rolling Forecast

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/multi_variable/best_model.pt \
    --data data/test_data.nc \
    --output-dir predictions/rolling_3steps \
    --rolling-steps 3 \
    --visualize
```

### Extended Rolling Forecast

```bash
# Maximum 6 steps (3 days)
python run_inference_convlstm.py \
    --checkpoint checkpoints/multi_variable/best_model.pt \
    --data data/test_data.nc \
    --output-dir predictions/rolling_6steps \
    --rolling-steps 6 \
    --batch-size 8 \
    --visualize
```

### Rolling Forecast Limitations

- **Maximum 6 steps**: Limited to 3 days ahead (12-hour intervals)
- **Multi-variable only**: Requires 56 output channels
- **Single-stream only**: Dual-stream models not yet supported (requires upstream future data)
- **Error accumulation**: Prediction quality degrades with longer horizons

## Evaluation

### Single-Step Evaluation

Evaluate multi-variable predictions:

```bash
python examples/evaluate_multi_variable.py \
    --predictions predictions/multi_variable/predictions.nc \
    --ground-truth data/test_data.nc \
    --output-dir predictions/multi_variable/metrics
```

### Rolling Forecast Evaluation

Evaluate rolling forecasts with per-timestep metrics:

```bash
python examples/evaluate_multi_variable.py \
    --predictions predictions/rolling_6steps/predictions.nc \
    --ground-truth data/test_data.nc \
    --output-dir predictions/rolling_6steps/metrics \
    --rolling-forecast
```

### Metrics Computed

For each variable and timestep:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Bias**: Mean prediction error
- **Correlation**: Spatial correlation coefficient

For precipitation specifically:
- **High Precipitation Accuracy**: Accuracy for events >10mm
- **Extreme Precipitation Accuracy**: Accuracy for events >50mm

See `MULTI_VARIABLE_EVALUATION.md` for detailed evaluation documentation.

## Model Architecture

### Output Layer Configuration

The only architectural change in multi-variable mode is the output layer:

```python
# Single-variable mode (default)
output_channels = 1  # Precipitation only

# Multi-variable mode
output_channels = 56  # 5 vars × 11 levels + precipitation
```

The encoder-decoder architecture remains unchanged, ensuring:
- Same memory footprint
- Same computational cost
- Same training dynamics

### Supported Model Variants

All four model variants support multi-variable mode:

1. **ConvLSTMUNet** (`model.py`): Base U-Net architecture
2. **DualStreamConvLSTMUNet** (`model_dual_stream.py`): Dual-stream with upstream region
3. **DeepConvLSTMUNet** (`model_deep.py`): Deeper architecture with more layers
4. **DeepDualStreamConvLSTMUNet** (`model_dual_stream_deep.py`): Deep dual-stream

## Loss Function

### Multi-Variable Loss Decomposition

The loss function combines precipitation and atmospheric variable losses:

```
Total Loss = precip_weight × precip_loss + atmos_loss
```

Where:
- **precip_loss**: Event-weighted MSE with latitude correction
- **atmos_loss**: Standard MSE across all 55 atmospheric channels
- **precip_weight**: Configurable multiplier (default: 10.0)

### Precipitation Loss (Event-Weighted)

Precipitation loss applies higher weights to significant events:

```python
# Base weight
weight = 1.0

# High precipitation (>10mm)
if precip > high_threshold:
    weight *= high_precip_weight  # default: 5.0

# Extreme precipitation (>50mm)
if precip > extreme_threshold:
    weight *= extreme_precip_weight  # default: 10.0

# Latitude correction (area weighting)
weight *= cos(latitude)
```

### Atmospheric Variable Loss

Simple MSE across all 55 atmospheric channels:

```python
atmos_loss = MSE(pred[:, :55], target[:, :55])
```

### Loss Weighting Rationale

The default 10:1 ratio (precipitation:atmospheric) ensures:
1. Precipitation remains the primary prediction target
2. Atmospheric variables provide physical constraints
3. Model learns meaningful atmospheric dynamics
4. Backward compatibility with single-variable performance

## Data Format

### Input Format

Input data remains unchanged from single-variable mode:
- **Shape**: `[B, T, C, H, W]`
- **Channels**: 56 (5 vars × 11 levels + precipitation)
- **Window**: T = 6 timesteps (3 days)

### Output Format

#### Single-Variable Mode
- **Shape**: `[B, 1, H, W]`
- **Channels**: 1 (precipitation only)

#### Multi-Variable Mode
- **Shape**: `[B, 56, H, W]`
- **Channels**: 56 (all variables)

### Channel Ordering

Channels 0-54: Atmospheric variables
- 0-10: DPT at 11 pressure levels
- 11-21: GPH at 11 pressure levels
- 22-32: TEM at 11 pressure levels
- 33-43: U at 11 pressure levels
- 44-54: V at 11 pressure levels

Channel 55: Precipitation

### Normalization

- **Atmospheric variables**: Z-score normalization (mean=0, std=1)
- **Precipitation**: log1p transformation + Z-score normalization

The same normalization is applied to both inputs and targets.

## Backward Compatibility

Multi-variable mode is fully backward compatible:

### Default Behavior

Omitting `--multi-variable` flag trains in single-variable mode:

```bash
# Single-variable mode (default)
python train_convlstm.py --data data.nc --output-dir checkpoints/single
```

### Checkpoint Compatibility

- Single-variable checkpoints load correctly (1 output channel)
- Multi-variable checkpoints load correctly (56 output channels)
- Mode is automatically detected from checkpoint
- No manual configuration needed

### Existing Scripts

All existing training and inference scripts work without modification:

```bash
# Existing script (single-variable)
python train_convlstm.py --data data.nc --output-dir checkpoints/old

# Still works exactly as before
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: CUDA out of memory error during training

**Solutions**:
```bash
# Reduce batch size
--batch-size 2

# Use gradient accumulation
--gradient-accumulation-steps 4

# Use smaller model
--hidden-channels 32 64

# Reduce rollout steps
--max-rollout-steps 2
```

#### 2. Rolling Forecast Not Supported

**Symptoms**: Error when trying rolling forecast with single-variable model

**Solution**: Train model with `--multi-variable` flag:
```bash
python train_convlstm.py --multi-variable --data data.nc --output-dir checkpoints/mv
```

#### 3. Dual-Stream Rolling Forecast Error

**Symptoms**: NotImplementedError for dual-stream rolling forecast

**Explanation**: Dual-stream models require upstream future data, which isn't available during rolling forecasts

**Solution**: Use single-stream model for rolling forecasts:
```bash
python train_convlstm.py --multi-variable --model-type base
```

#### 4. Loss Imbalance

**Symptoms**: Precipitation accuracy poor despite low overall loss

**Solution**: Increase precipitation loss weight:
```bash
--precip-loss-weight 15.0  # or higher
```

#### 5. Training Instability with Rolling

**Symptoms**: NaN losses or exploding gradients during rolling training

**Solutions**:
```bash
# Enable gradient clipping
--gradient-clip-norm 1.0

# Reduce rollout steps
--max-rollout-steps 2

# Use smaller learning rate
--learning-rate 5e-4
```

### Performance Tips

1. **Use Mixed Precision**: Always enable `--use-amp` for faster training
2. **Batch Size**: Start with 4, reduce if OOM occurs
3. **Gradient Accumulation**: Simulate larger batches without memory cost
4. **Checkpoint Frequency**: Save every 10 epochs to avoid losing progress
5. **Validation Frequency**: Check every 500 steps for early stopping

### Debugging

Enable debug logging for detailed information:

```bash
python train_convlstm.py \
    --multi-variable \
    --log-level DEBUG \
    --data data.nc \
    --output-dir checkpoints/debug
```

Check training logs:
```bash
tail -f checkpoints/debug/training.log
```

## Example Workflows

### Complete Training Pipeline

```bash
# 1. Train multi-variable model
python train_convlstm.py \
    --data data/train.nc \
    --output-dir checkpoints/multi_variable \
    --multi-variable \
    --batch-size 4 \
    --num-epochs 100 \
    --use-amp

# 2. Generate predictions
python run_inference_convlstm.py \
    --checkpoint checkpoints/multi_variable/best_model.pt \
    --data data/test.nc \
    --output-dir predictions/multi_variable \
    --visualize

# 3. Evaluate results
python examples/evaluate_multi_variable.py \
    --predictions predictions/multi_variable/predictions.nc \
    --ground-truth data/test.nc \
    --output-dir predictions/multi_variable/metrics
```

### Rolling Forecast Pipeline

```bash
# 1. Train with rolling supervision
python train_convlstm.py \
    --data data/train.nc \
    --output-dir checkpoints/rolling \
    --multi-variable \
    --enable-rollout-training \
    --max-rollout-steps 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --num-epochs 100 \
    --use-amp

# 2. Generate rolling forecasts
python run_inference_convlstm.py \
    --checkpoint checkpoints/rolling/best_model.pt \
    --data data/test.nc \
    --output-dir predictions/rolling \
    --rolling-steps 6 \
    --visualize

# 3. Evaluate rolling forecasts
python examples/evaluate_multi_variable.py \
    --predictions predictions/rolling/predictions.nc \
    --ground-truth data/test.nc \
    --output-dir predictions/rolling/metrics \
    --rolling-forecast
```

### Comparison Experiment

```bash
# Train single-variable baseline
python train_convlstm.py \
    --data data/train.nc \
    --output-dir checkpoints/single_variable \
    --batch-size 4 \
    --num-epochs 100

# Train multi-variable model
python train_convlstm.py \
    --data data/train.nc \
    --output-dir checkpoints/multi_variable \
    --multi-variable \
    --batch-size 4 \
    --num-epochs 100

# Compare results
python examples/compare_models.py \
    --model1 checkpoints/single_variable/best_model.pt \
    --model2 checkpoints/multi_variable/best_model.pt \
    --data data/test.nc \
    --output-dir comparisons/single_vs_multi
```

## References

- **Design Document**: `.kiro/specs/multi-variable-rolling-forecast/design.md`
- **Requirements**: `.kiro/specs/multi-variable-rolling-forecast/requirements.md`
- **Tasks**: `.kiro/specs/multi-variable-rolling-forecast/tasks.md`
- **Evaluation Guide**: `convlstm/MULTI_VARIABLE_EVALUATION.md`
- **Rolling Forecast Documentation**: `docs/Rolling_forecast.md`
- **Training Guide**: `convlstm/TRAINING_GUIDE.md`

## Additional Resources

### Example Scripts

- `examples/train_multi_variable.sh`: Comprehensive training examples
- `examples/rolling_forecast.sh`: Rolling forecast examples
- `examples/evaluate_multi_variable.py`: Evaluation script

### Documentation

- `convlstm/README.md`: Module overview
- `convlstm/TRAINING_GUIDE.md`: Detailed training documentation
- `convlstm/INFERENCE_GUIDE.md`: Inference documentation
- `docs/Rolling_forecast.md`: Rolling forecast theory and implementation
