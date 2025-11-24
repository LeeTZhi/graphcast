# ConvLSTM Scripts Overview

This document provides an overview of the main scripts for training and inference with ConvLSTM models.

## Scripts

### 1. `train_convlstm.py` - Training Script

**Purpose**: Train ConvLSTM models for weather prediction

**Location**: Root directory

**Key Features**:
- Configurable model architecture (hidden channels, kernel size)
- Support for upstream region inclusion
- Memory optimization (mixed precision, gradient accumulation)
- Checkpoint saving and resumption
- Comprehensive logging

**Basic Usage**:
```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline
```

**Documentation**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### 2. `run_inference_convlstm.py` - Inference Script

**Purpose**: Generate predictions with trained ConvLSTM models

**Location**: Root directory

**Key Features**:
- Load trained models from checkpoints
- Batch prediction on test data
- Save predictions to NetCDF
- Generate visualizations (maps, comparisons, error plots)
- Compare two experiments side-by-side

**Basic Usage**:
```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --normalizer checkpoints/baseline/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/baseline
```

**Documentation**: See [INFERENCE_CLI_GUIDE.md](INFERENCE_CLI_GUIDE.md)

## Workflow

### Complete Training and Inference Workflow

```
1. Train baseline model (downstream only)
   ↓
2. Train comparison model (with upstream)
   ↓
3. Run inference on both models
   ↓
4. Compare results with visualizations
```

### Step-by-Step Example

#### Step 1: Train Baseline Model

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --hidden-channels 32 64 \
    --batch-size 4 \
    --num-epochs 50
```

**Outputs**:
- `checkpoints/baseline/best_model.pt`
- `checkpoints/baseline/normalizer.pkl`
- `checkpoints/baseline/training.log`

#### Step 2: Train Comparison Model (With Upstream)

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/with_upstream \
    --include-upstream \
    --hidden-channels 32 64 \
    --batch-size 4 \
    --num-epochs 50
```

**Outputs**:
- `checkpoints/with_upstream/best_model.pt`
- `checkpoints/with_upstream/normalizer.pkl`
- `checkpoints/with_upstream/training.log`

#### Step 3: Run Inference on Baseline Model

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --normalizer checkpoints/baseline/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/baseline \
    --visualize
```

**Outputs**:
- `outputs/baseline/predictions.nc`
- `outputs/baseline/visualizations/*.png`
- `outputs/baseline/inference.log`

#### Step 4: Compare Both Models

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --normalizer checkpoints/baseline/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/comparison \
    --compare-checkpoint checkpoints/with_upstream/best_model.pt \
    --compare-normalizer checkpoints/with_upstream/normalizer.pkl \
    --compare-upstream \
    --exp1-name "Baseline" \
    --exp2-name "With_Upstream" \
    --visualize \
    --viz-timesteps 0 5 10 15 20
```

**Outputs**:
- `outputs/comparison/predictions.nc`
- `outputs/comparison/comparison_predictions.nc`
- `outputs/comparison/visualizations/comparison_*.png`
- `outputs/comparison/visualizations/error_comparison_*.png`
- `outputs/comparison/inference.log`

## Common Arguments

### Training Script

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to training data | Required |
| `--output-dir` | Checkpoint directory | Required |
| `--include-upstream` | Use upstream region | False |
| `--hidden-channels` | Model hidden dimensions | [32, 64] |
| `--batch-size` | Training batch size | 4 |
| `--learning-rate` | Initial learning rate | 1e-3 |
| `--num-epochs` | Number of epochs | 100 |
| `--use-amp` | Mixed precision training | True |

### Inference Script

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Model checkpoint path | Required |
| `--normalizer` | Normalizer path | Required |
| `--data` | Input data path | Required |
| `--output-dir` | Output directory | Required |
| `--include-upstream` | Use upstream region | False |
| `--visualize` | Generate plots | False |
| `--viz-timesteps` | Timesteps to visualize | First 5 |
| `--compare-checkpoint` | Second model for comparison | None |

## File Organization

### Training Outputs

```
checkpoints/
├── baseline/
│   ├── best_model.pt              # Best model checkpoint
│   ├── normalizer.pkl             # Data normalizer
│   ├── checkpoint_step_1000.pt    # Periodic checkpoints
│   ├── checkpoint_step_2000.pt
│   └── training.log               # Training log
└── with_upstream/
    ├── best_model.pt
    ├── normalizer.pkl
    └── training.log
```

### Inference Outputs

```
outputs/
├── baseline/
│   ├── predictions.nc             # Predictions
│   ├── inference.log              # Inference log
│   └── visualizations/            # Visualization plots
│       ├── Baseline_prediction_0.png
│       ├── Baseline_error_0.png
│       └── ...
└── comparison/
    ├── predictions.nc             # Experiment 1 predictions
    ├── comparison_predictions.nc  # Experiment 2 predictions
    ├── inference.log
    └── visualizations/
        ├── comparison_comparison_0.png
        ├── error_comparison_multi_error_0.png
        └── ...
```

## Tips

### Memory Management

**Training**:
- Start with `--batch-size 4` for 12GB GPU
- Use `--gradient-accumulation-steps 2` to simulate larger batches
- Enable `--use-amp` for mixed precision (default: on)

**Inference**:
- Use `--batch-size 8` for faster inference
- Reduce to `--batch-size 4` if memory issues occur
- Use `--device cpu` if no GPU available

### Experiment Organization

Use descriptive directory names:
```bash
# Training
--output-dir checkpoints/baseline_32_64_lr1e3
--output-dir checkpoints/upstream_32_64_lr1e3

# Inference
--output-dir outputs/baseline_eval_test_set
--output-dir outputs/comparison_baseline_vs_upstream
```

### Visualization

For quick inspection:
```bash
--visualize --viz-timesteps 0 5 10
```

For publication-quality figures:
```bash
--visualize --viz-format pdf --viz-dpi 300
```

## See Also

- [README.md](README.md) - ConvLSTM module overview
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training guide
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Programmatic inference API
- [INFERENCE_CLI_GUIDE.md](INFERENCE_CLI_GUIDE.md) - CLI inference guide
