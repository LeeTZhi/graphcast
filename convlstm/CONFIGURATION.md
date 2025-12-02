# ConvLSTM Configuration Parameters

This document provides a comprehensive reference for all configuration parameters in the ConvLSTM weather prediction system.

## Table of Contents

- [Multi-Variable Parameters](#multi-variable-parameters)
- [Model Architecture](#model-architecture)
- [Training Hyperparameters](#training-hyperparameters)
- [Data Processing](#data-processing)
- [Loss Function](#loss-function)
- [Memory Optimization](#memory-optimization)
- [Device and Hardware](#device-and-hardware)
- [Checkpointing](#checkpointing)
- [Logging](#logging)
- [Region Configuration](#region-configuration)

## Multi-Variable Parameters

### `--multi-variable`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable multi-variable prediction mode (56 output channels)
- **Usage**: `--multi-variable`
- **Notes**: 
  - Predicts all atmospheric variables plus precipitation
  - Required for rolling forecasts
  - Backward compatible (omit for single-variable mode)

### `--precip-loss-weight`
- **Type**: Float
- **Default**: `10.0`
- **Range**: `1.0` - `20.0` (typical)
- **Description**: Weight multiplier for precipitation loss vs atmospheric variables
- **Usage**: `--precip-loss-weight 15.0`
- **Notes**:
  - Higher values prioritize precipitation accuracy
  - Lower values give more weight to atmospheric variables
  - Only applies in multi-variable mode

### `--max-rollout-steps`
- **Type**: Integer
- **Default**: `6`
- **Range**: `1` - `6`
- **Description**: Maximum number of rolling forecast steps
- **Usage**: `--max-rollout-steps 4`
- **Notes**:
  - Each step = 12 hours (6 steps = 3 days)
  - Only applies in multi-variable mode
  - Longer horizons accumulate more error

### `--enable-rollout-training`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable autoregressive training for rolling forecasts
- **Usage**: `--enable-rollout-training`
- **Notes**:
  - Requires more memory (use gradient accumulation)
  - Improves multi-step prediction accuracy
  - Requires `--multi-variable` flag

## Model Architecture

### `--model-type`
- **Type**: String (choice)
- **Default**: `base`
- **Choices**: `base`, `deep`, `dual_stream`, `dual_stream_deep`
- **Description**: Model architecture variant
- **Usage**: `--model-type deep`
- **Notes**:
  - `base`: Standard ConvLSTM U-Net
  - `deep`: Deeper architecture with more layers
  - `dual_stream`: Includes upstream region
  - `dual_stream_deep`: Deep architecture with dual-stream

### `--hidden-channels`
- **Type**: List of integers
- **Default**: `[32, 64]`
- **Description**: Hidden channel dimensions for encoder layers
- **Usage**: `--hidden-channels 64 128`
- **Notes**:
  - More channels = more capacity but more memory
  - Typical values: `[16, 32]`, `[32, 64]`, `[64, 128]`
  - Affects memory usage significantly

### `--kernel-size`
- **Type**: Integer
- **Default**: `3`
- **Range**: `3`, `5`, `7` (odd numbers)
- **Description**: Convolution kernel size
- **Usage**: `--kernel-size 5`
- **Notes**:
  - Larger kernels = larger receptive field
  - Larger kernels = more parameters and memory
  - 3 is usually sufficient

### `--input-channels`
- **Type**: Integer
- **Default**: `56`
- **Description**: Number of input channels
- **Usage**: `--input-channels 56`
- **Notes**:
  - Should match data format (5 vars × 11 levels + precipitation)
  - Rarely needs to be changed

## Training Hyperparameters

### `--learning-rate`
- **Type**: Float
- **Default**: `1e-3` (0.001)
- **Range**: `1e-5` - `1e-2`
- **Description**: Initial learning rate for optimizer
- **Usage**: `--learning-rate 5e-4`
- **Notes**:
  - Lower values = more stable but slower training
  - Higher values = faster but may be unstable
  - Typical range: `5e-4` to `1e-3`

### `--batch-size`
- **Type**: Integer
- **Default**: `4`
- **Range**: `1` - `16` (depends on GPU memory)
- **Description**: Number of samples per batch
- **Usage**: `--batch-size 8`
- **Notes**:
  - Larger batches = more stable gradients but more memory
  - Reduce if out of memory
  - Use gradient accumulation to simulate larger batches

### `--num-epochs`
- **Type**: Integer
- **Default**: `100`
- **Description**: Number of training epochs
- **Usage**: `--num-epochs 50`
- **Notes**:
  - More epochs = better convergence but longer training
  - Use early stopping to prevent overfitting
  - Typical range: 50-200

### `--weight-decay`
- **Type**: Float
- **Default**: `0.0`
- **Range**: `0.0` - `1e-3`
- **Description**: L2 regularization weight
- **Usage**: `--weight-decay 1e-4`
- **Notes**:
  - Helps prevent overfitting
  - Typical values: `1e-5` to `1e-4`
  - 0 = no regularization

### `--gradient-clip-norm`
- **Type**: Float
- **Default**: `None` (no clipping)
- **Range**: `0.1` - `10.0`
- **Description**: Maximum gradient norm for clipping
- **Usage**: `--gradient-clip-norm 1.0`
- **Notes**:
  - Prevents exploding gradients
  - Essential for rolling forecast training
  - Typical values: 0.5 to 2.0

### `--gradient-accumulation-steps`
- **Type**: Integer
- **Default**: `1`
- **Description**: Number of steps to accumulate gradients
- **Usage**: `--gradient-accumulation-steps 4`
- **Notes**:
  - Simulates larger batch size without memory cost
  - Effective batch size = batch_size × accumulation_steps
  - Useful for memory-constrained GPUs

## Data Processing

### `--data`
- **Type**: String (file path)
- **Required**: Yes
- **Description**: Path to input NetCDF data file
- **Usage**: `--data data/regional_weather.nc`
- **Notes**:
  - Must be xarray-compatible NetCDF format
  - Should contain all required variables

### `--window-size`
- **Type**: Integer
- **Default**: `6`
- **Description**: Number of historical timesteps in input window
- **Usage**: `--window-size 12`
- **Notes**:
  - 6 timesteps = 3 days at 12-hour intervals
  - Larger windows = more context but more memory
  - Must match model training configuration

### `--target-offset`
- **Type**: Integer
- **Default**: `1`
- **Description**: Number of timesteps ahead to predict
- **Usage**: `--target-offset 2`
- **Notes**:
  - 1 = predict 12 hours ahead
  - 2 = predict 24 hours ahead
  - Typically kept at 1

### `--train-ratio`
- **Type**: Float
- **Default**: `0.7`
- **Range**: `0.5` - `0.9`
- **Description**: Fraction of data for training
- **Usage**: `--train-ratio 0.8`

### `--val-ratio`
- **Type**: Float
- **Default**: `0.15`
- **Range**: `0.1` - `0.3`
- **Description**: Fraction of data for validation
- **Usage**: `--val-ratio 0.2`

### `--num-workers`
- **Type**: Integer
- **Default**: `4`
- **Range**: `0` - `8`
- **Description**: Number of data loading workers
- **Usage**: `--num-workers 2`
- **Notes**:
  - More workers = faster data loading
  - Reduce if memory issues occur
  - 0 = load in main process

## Loss Function

### `--high-precip-threshold`
- **Type**: Float
- **Default**: `10.0`
- **Description**: Threshold (mm) for high precipitation events
- **Usage**: `--high-precip-threshold 15.0`
- **Notes**:
  - Events above threshold get higher weight
  - Helps model focus on significant rainfall

### `--high-precip-weight`
- **Type**: Float
- **Default**: `5.0`
- **Description**: Weight multiplier for high precipitation events
- **Usage**: `--high-precip-weight 7.0`
- **Notes**:
  - Higher values = more focus on high precipitation
  - Typical range: 3.0 to 10.0

### `--extreme-precip-threshold`
- **Type**: Float
- **Default**: `50.0`
- **Description**: Threshold (mm) for extreme precipitation events
- **Usage**: `--extreme-precip-threshold 75.0`

### `--extreme-precip-weight`
- **Type**: Float
- **Default**: `10.0`
- **Description**: Weight multiplier for extreme precipitation events
- **Usage**: `--extreme-precip-weight 15.0`

## Memory Optimization

### `--use-amp`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Enable automatic mixed precision training
- **Usage**: `--use-amp`
- **Notes**:
  - Reduces memory usage by ~40%
  - Speeds up training on modern GPUs
  - Highly recommended for all training

### `--no-amp`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Explicitly disable mixed precision
- **Usage**: `--no-amp`
- **Notes**:
  - Use if AMP causes numerical issues
  - Required for CPU training

## Device and Hardware

### `--device`
- **Type**: String (choice)
- **Default**: `auto` (automatic detection)
- **Choices**: `auto`, `cuda`, `mps`, `cpu`
- **Description**: Device to use for training/inference
- **Usage**: `--device cuda`
- **Notes**:
  - `auto`: Automatically selects best available device
  - `cuda`: NVIDIA GPU
  - `mps`: Apple Silicon GPU (M1/M2/M3)
  - `cpu`: CPU only (slow)

## Checkpointing

### `--output-dir`
- **Type**: String (directory path)
- **Required**: Yes
- **Description**: Directory to save checkpoints and logs
- **Usage**: `--output-dir checkpoints/experiment1`
- **Notes**:
  - Created automatically if doesn't exist
  - Contains: best_model.pt, checkpoints, logs, normalizer

### `--checkpoint-frequency`
- **Type**: Integer
- **Default**: `10`
- **Description**: Save checkpoint every N epochs
- **Usage**: `--checkpoint-frequency 5`

### `--resume`
- **Type**: String (file path)
- **Default**: `None`
- **Description**: Path to checkpoint to resume training from
- **Usage**: `--resume checkpoints/exp1/checkpoint_epoch_50.pt`

### `--early-stopping-patience`
- **Type**: Integer
- **Default**: `20`
- **Description**: Stop training if no improvement for N epochs
- **Usage**: `--early-stopping-patience 15`

## Logging

### `--log-level`
- **Type**: String (choice)
- **Default**: `INFO`
- **Choices**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- **Description**: Logging verbosity level
- **Usage**: `--log-level DEBUG`

### `--validation-frequency`
- **Type**: Integer
- **Default**: `500`
- **Description**: Run validation every N training steps
- **Usage**: `--validation-frequency 100`

## Region Configuration

### `--include-upstream`
- **Type**: Flag (boolean)
- **Default**: `False`
- **Description**: Include upstream region in input
- **Usage**: `--include-upstream`
- **Notes**:
  - Enables comparative experiments
  - Requires dual-stream model

### `--downstream-lat-min`
- **Type**: Float
- **Default**: `25.0`
- **Description**: Minimum latitude for downstream region
- **Usage**: `--downstream-lat-min 30.0`

### `--downstream-lat-max`
- **Type**: Float
- **Default**: `40.0`
- **Description**: Maximum latitude for downstream region
- **Usage**: `--downstream-lat-max 45.0`

### `--downstream-lon-min`
- **Type**: Float
- **Default**: `110.0`
- **Description**: Minimum longitude for downstream region
- **Usage**: `--downstream-lon-min 105.0`

### `--downstream-lon-max`
- **Type**: Float
- **Default**: `125.0`
- **Description**: Maximum longitude for downstream region
- **Usage**: `--downstream-lon-max 120.0`

### `--upstream-lat-min`
- **Type**: Float
- **Default**: `25.0`
- **Description**: Minimum latitude for upstream region
- **Usage**: `--upstream-lat-min 30.0`

### `--upstream-lat-max`
- **Type**: Float
- **Default**: `50.0`
- **Description**: Maximum latitude for upstream region
- **Usage**: `--upstream-lat-max 45.0`

### `--upstream-lon-min`
- **Type**: Float
- **Default**: `70.0`
- **Description**: Minimum longitude for upstream region
- **Usage**: `--upstream-lon-min 85.0`

### `--upstream-lon-max`
- **Type**: Float
- **Default**: `110.0`
- **Description**: Maximum longitude for upstream region
- **Usage**: `--upstream-lon-max 105.0`

## Example Configurations

### Minimal Configuration

```bash
python train_convlstm.py \
    --data data/weather.nc \
    --output-dir checkpoints/minimal
```

### Recommended Configuration

```bash
python train_convlstm.py \
    --data data/weather.nc \
    --output-dir checkpoints/recommended \
    --batch-size 4 \
    --num-epochs 100 \
    --use-amp \
    --gradient-clip-norm 1.0
```

### Multi-Variable Configuration

```bash
python train_convlstm.py \
    --data data/weather.nc \
    --output-dir checkpoints/multi_variable \
    --multi-variable \
    --precip-loss-weight 10.0 \
    --batch-size 4 \
    --num-epochs 100 \
    --use-amp
```

### Rolling Forecast Training

```bash
python train_convlstm.py \
    --data data/weather.nc \
    --output-dir checkpoints/rolling \
    --multi-variable \
    --enable-rollout-training \
    --max-rollout-steps 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --gradient-clip-norm 1.0 \
    --use-amp
```

### Memory-Optimized Configuration

```bash
python train_convlstm.py \
    --data data/weather.nc \
    --output-dir checkpoints/memory_opt \
    --multi-variable \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --hidden-channels 32 64 \
    --num-workers 2 \
    --use-amp
```

### High-Capacity Configuration

```bash
python train_convlstm.py \
    --data data/weather.nc \
    --output-dir checkpoints/high_capacity \
    --multi-variable \
    --model-type deep \
    --hidden-channels 64 128 \
    --batch-size 8 \
    --learning-rate 5e-4 \
    --num-epochs 200 \
    --use-amp
```

## Configuration Files

You can also use YAML configuration files:

```yaml
# config.yaml
data: data/weather.nc
output_dir: checkpoints/from_config

# Multi-variable settings
multi_variable: true
precip_loss_weight: 10.0
max_rollout_steps: 6

# Model architecture
model_type: base
hidden_channels: [32, 64]
kernel_size: 3

# Training hyperparameters
batch_size: 4
learning_rate: 0.001
num_epochs: 100
weight_decay: 0.0001
gradient_clip_norm: 1.0

# Memory optimization
use_amp: true
gradient_accumulation_steps: 1

# Device
device: auto
```

Load with:
```bash
python train_convlstm.py --config config.yaml
```

## See Also

- **Multi-Variable Guide**: `convlstm/MULTI_VARIABLE_GUIDE.md`
- **Training Guide**: `convlstm/TRAINING_GUIDE.md`
- **Module README**: `convlstm/README.md`
- **Example Scripts**: `examples/train_multi_variable.sh`, `examples/rolling_forecast.sh`
