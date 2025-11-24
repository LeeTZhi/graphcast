# ConvLSTM Training Guide

This guide explains how to use the `train_convlstm.py` script to train ConvLSTM weather prediction models.

## Quick Start

### Basic Training (Downstream Only)

Train a baseline model using only the downstream region:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline
```

### Training with Upstream Region

Train a model that includes the upstream region for comparative experiments:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/with_upstream \
    --include-upstream
```

## Configuration Options

### Data Configuration

- `--data`: Path to NetCDF data file (required)
- `--output-dir`: Directory to save checkpoints and logs (required)

### Region Configuration

- `--include-upstream`: Include upstream region in input (default: False)
- `--downstream-lat-min/max`: Downstream region latitude bounds (default: 25.0-40.0)
- `--downstream-lon-min/max`: Downstream region longitude bounds (default: 110.0-125.0)
- `--upstream-lat-min/max`: Upstream region latitude bounds (default: 25.0-50.0)
- `--upstream-lon-min/max`: Upstream region longitude bounds (default: 70.0-110.0)

### Model Architecture

- `--hidden-channels`: Hidden channel dimensions (default: 32 64)
  - Example: `--hidden-channels 64 128` for larger model
- `--kernel-size`: Convolutional kernel size (default: 3)

### Training Configuration

- `--learning-rate`: Initial learning rate (default: 1e-3)
- `--batch-size`: Batch size (default: 4)
- `--num-epochs`: Number of training epochs (default: 100)
- `--gradient-clip-norm`: Gradient clipping norm (default: 1.0)
- `--weight-decay`: Weight decay for AdamW (default: 1e-5)

### Data Processing

- `--window-size`: Number of historical timesteps (default: 6)
- `--target-offset`: Timesteps ahead to predict (default: 1)
- `--train-ratio`: Fraction for training (default: 0.7)
- `--val-ratio`: Fraction for validation (default: 0.15)

### Loss Function

- `--high-precip-threshold`: Threshold for high precipitation (default: 10.0 mm)
- `--high-precip-weight`: Weight multiplier for high precipitation (default: 3.0)

### Memory Optimization

- `--use-amp`: Enable automatic mixed precision (default: True)
- `--no-amp`: Disable automatic mixed precision
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: 1)
- `--num-workers`: DataLoader worker processes (default: 2)

### Checkpointing

- `--checkpoint-frequency`: Save checkpoint every N steps (default: 1000)
- `--validation-frequency`: Run validation every N steps (default: 500)
- `--early-stopping-patience`: Stop after N epochs without improvement (default: 10)
- `--resume`: Path to checkpoint to resume from

### Logging and Device

- `--log-level`: Logging level (default: INFO)
- `--device`: Device to use (default: auto, choices: auto, cuda, mps, cpu)
  - `auto`: Automatically select best available device (cuda > mps > cpu)
  - `cuda`: Use NVIDIA GPU
  - `mps`: Use Apple Silicon GPU (Mac M1/M2/M3)
  - `cpu`: Use CPU only

## Common Use Cases

### Memory-Optimized Training (12GB GPU)

For training on GPUs with limited memory:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/memory_opt \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --use-amp \
    --hidden-channels 32 64
```

### Custom Hyperparameters

Train with custom model and training settings:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/custom \
    --hidden-channels 64 128 \
    --batch-size 8 \
    --learning-rate 5e-4 \
    --num-epochs 50 \
    --window-size 12
```

### Resume Training

Resume training from a saved checkpoint:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --resume checkpoints/baseline/checkpoint_epoch_10.pt
```

### Comparative Experiments

Train two models for comparison:

```bash
# Experiment 1: Baseline (downstream only)
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/exp1_baseline

# Experiment 2: With upstream region
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/exp2_upstream \
    --include-upstream
```

## Output Files

The training script creates the following files in the output directory:

- `training.log`: Detailed training logs
- `best_model.pt`: Best model checkpoint (lowest validation loss)
- `checkpoint_epoch_N.pt`: Periodic checkpoints every 10 epochs
- `normalizer.pkl`: Data normalization statistics
- `interrupted_checkpoint.pt`: Checkpoint saved if training is interrupted (Ctrl+C)

## Checkpoint Contents

Each checkpoint file contains:

- Model state (parameters)
- Optimizer state
- Learning rate scheduler state
- Training progress (epoch, step, best validation loss)
- Model configuration
- Region configuration
- Version metadata

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce batch size: `--batch-size 2`
2. Increase gradient accumulation: `--gradient-accumulation-steps 4`
3. Reduce model size: `--hidden-channels 16 32`
4. Ensure mixed precision is enabled: `--use-amp`

### NaN Loss

If loss becomes NaN during training:

1. Reduce learning rate: `--learning-rate 5e-4`
2. Increase gradient clipping: `--gradient-clip-norm 0.5`
3. Check for invalid data (NaN/Inf in inputs)

### Slow Training

If training is too slow:

1. Increase number of workers: `--num-workers 4`
2. Ensure GPU is being used: check logs for "Device: cuda"
3. Reduce validation frequency: `--validation-frequency 1000`

## Monitoring Training

The script logs comprehensive information during training:

- Model architecture and parameter count
- GPU information (if available)
- Data loading and splitting
- Training progress (loss, learning rate, time per epoch)
- Validation results
- Checkpoint saving

All logs are saved to `training.log` in the output directory and displayed on the console.

## Example Training Session

```bash
$ python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/test \
    --batch-size 4 \
    --num-epochs 10

2024-01-15 10:00:00 - INFO - ================================================================================
2024-01-15 10:00:00 - INFO - ConvLSTM Weather Prediction Training
2024-01-15 10:00:00 - INFO - ================================================================================
2024-01-15 10:00:00 - INFO - Configuration:
2024-01-15 10:00:00 - INFO -   Data: data/regional_weather.nc
2024-01-15 10:00:00 - INFO -   Output directory: checkpoints/test
2024-01-15 10:00:00 - INFO -   Include upstream: False
2024-01-15 10:00:00 - INFO -   Hidden channels: [32, 64]
2024-01-15 10:00:00 - INFO -   Batch size: 4
2024-01-15 10:00:00 - INFO -   Learning rate: 0.001
2024-01-15 10:00:00 - INFO -   Number of epochs: 10
2024-01-15 10:00:00 - INFO -   Mixed precision: True
2024-01-15 10:00:00 - INFO -   Device: cuda
2024-01-15 10:00:00 - INFO -   GPU: NVIDIA GeForce RTX 3060
2024-01-15 10:00:00 - INFO -   GPU memory: 12.00 GB
...
2024-01-15 10:05:00 - INFO - Epoch 1/10: train_loss=0.5234, val_loss=0.4567, lr=0.001000, time=45.23s
2024-01-15 10:10:00 - INFO - Saved best model with val_loss=0.4567
...
2024-01-15 11:00:00 - INFO - ================================================================================
2024-01-15 11:00:00 - INFO - Training completed successfully!
2024-01-15 11:00:00 - INFO - Best validation loss: 0.3456
2024-01-15 11:00:00 - INFO - Total training time: 3600.00s
2024-01-15 11:00:00 - INFO - Checkpoints saved to: checkpoints/test
```
