# ConvLSTM Configuration Files

This directory contains example YAML configuration files for different training scenarios. These configurations can be used as templates or loaded directly by the training script.

## Available Configurations

### 1. baseline_12gb.yaml
**Purpose:** Standard baseline experiment for 12GB GPU  
**Use Case:** Training with downstream region only  
**Memory:** ~4-5 GB peak  
**Training Time:** ~50-75 hours for 100 epochs  

```bash
python train_convlstm.py --config convlstm/configs/baseline_12gb.yaml
```

### 2. upstream_12gb.yaml
**Purpose:** Comparative experiment with upstream region for 12GB GPU  
**Use Case:** Training with both upstream and downstream regions  
**Memory:** ~5-7 GB peak  
**Training Time:** ~65-100 hours for 100 epochs  

```bash
python train_convlstm.py --config convlstm/configs/upstream_12gb.yaml
```

### 3. baseline_8gb.yaml
**Purpose:** Memory-optimized baseline for 8GB GPU  
**Use Case:** Training on lower-end hardware  
**Memory:** ~2-3 GB peak  
**Training Time:** ~35-50 hours for 100 epochs  
**Note:** Smaller model, slightly lower accuracy  

```bash
python train_convlstm.py --config convlstm/configs/baseline_8gb.yaml
```

### 4. high_capacity.yaml
**Purpose:** Maximum accuracy model for 24GB+ GPU  
**Use Case:** Research with high-end hardware  
**Memory:** ~15-20 GB peak  
**Training Time:** ~150-225 hours for 150 epochs  
**Note:** Requires RTX 3090, A5000, or better  

```bash
python train_convlstm.py --config convlstm/configs/high_capacity.yaml
```

### 5. quick_test.yaml
**Purpose:** Rapid testing and debugging  
**Use Case:** Development, sanity checks, pipeline testing  
**Memory:** <1 GB peak  
**Training Time:** ~10-25 minutes for 5 epochs  
**Note:** Not for production use  

```bash
python train_convlstm.py --config convlstm/configs/quick_test.yaml
```

## Configuration Structure

All configuration files follow this structure:

```yaml
# Data Configuration
data:
  data_path: "path/to/data.nc"
  output_dir: "path/to/checkpoints"
  train_ratio: 0.7
  val_ratio: 0.15
  window_size: 6
  target_offset: 1

# Region Configuration
region:
  downstream:
    lat_min: 25.0
    lat_max: 40.0
    lon_min: 110.0
    lon_max: 125.0
  upstream:  # Optional
    lat_min: 25.0
    lat_max: 40.0
    lon_min: 70.0
    lon_max: 110.0
  include_upstream: false

# Model Architecture
model:
  input_channels: 56
  hidden_channels: [32, 64]
  kernel_size: 3
  output_channels: 1

# Training Configuration
training:
  learning_rate: 0.001
  weight_decay: 0.00001
  batch_size: 4
  num_epochs: 100
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  use_amp: true
  num_workers: 2
  pin_memory: true
  scheduler_type: "cosine"
  scheduler_patience: 5
  scheduler_factor: 0.5

# Loss Function Configuration
loss:
  high_precip_threshold: 10.0
  high_precip_weight: 3.0
  latitude_weighting: true

# Checkpointing and Validation
checkpointing:
  checkpoint_frequency: 1000
  validation_frequency: 500
  early_stopping_patience: 10
  save_best_only: false

# Logging
logging:
  log_level: "INFO"
  log_to_file: true
  log_to_console: true

# Device Configuration
device:
  device: "auto"
  cuda_device: 0
```

## Configuration Options Reference

### Data Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `data_path` | string | required | Path to input NetCDF file |
| `output_dir` | string | required | Directory for checkpoints and logs |
| `train_ratio` | float | 0.7 | Fraction of data for training |
| `val_ratio` | float | 0.15 | Fraction of data for validation |
| `window_size` | int | 6 | Number of historical timesteps |
| `target_offset` | int | 1 | Timesteps ahead to predict |

### Region Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `downstream.lat_min` | float | 25.0 | Minimum latitude of downstream region |
| `downstream.lat_max` | float | 40.0 | Maximum latitude of downstream region |
| `downstream.lon_min` | float | 110.0 | Minimum longitude of downstream region |
| `downstream.lon_max` | float | 125.0 | Maximum longitude of downstream region |
| `upstream.lat_min` | float | 25.0 | Minimum latitude of upstream region |
| `upstream.lat_max` | float | 40.0 | Maximum latitude of upstream region |
| `upstream.lon_min` | float | 70.0 | Minimum longitude of upstream region |
| `upstream.lon_max` | float | 110.0 | Maximum longitude of upstream region |
| `include_upstream` | bool | false | Whether to include upstream region |

### Model Architecture

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input_channels` | int | 56 | Number of input channels (5 vars × 11 levels + 1 precip) |
| `hidden_channels` | list[int] | [32, 64] | Hidden dimensions for encoder and bottleneck |
| `kernel_size` | int | 3 | Convolutional kernel size |
| `output_channels` | int | 1 | Number of output channels (precipitation) |

**Hidden Channels Guidelines:**
- `[8, 16]`: Quick testing, <1 GB memory
- `[16, 32]`: Low memory (8GB GPU), ~2-3 GB
- `[32, 64]`: Standard (12GB GPU), ~4-5 GB
- `[64, 128]`: High capacity (24GB GPU), ~10-15 GB
- `[64, 128, 256]`: Maximum capacity (24GB+ GPU), ~15-20 GB

### Training Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `learning_rate` | float | 0.001 | Initial learning rate |
| `weight_decay` | float | 0.00001 | L2 regularization weight |
| `batch_size` | int | 4 | Batch size per GPU |
| `num_epochs` | int | 100 | Total training epochs |
| `gradient_clip_norm` | float | 1.0 | Maximum gradient norm |
| `gradient_accumulation_steps` | int | 1 | Steps to accumulate gradients |
| `use_amp` | bool | true | Use automatic mixed precision |
| `num_workers` | int | 2 | DataLoader worker processes |
| `pin_memory` | bool | true | Pin memory for faster GPU transfer |
| `scheduler_type` | string | "cosine" | LR scheduler: "cosine", "step", or "none" |
| `scheduler_patience` | int | 5 | Patience for ReduceLROnPlateau |
| `scheduler_factor` | float | 0.5 | LR reduction factor |
| `warmup_epochs` | int | 0 | Warmup epochs (optional) |

**Effective Batch Size:**
```
effective_batch_size = batch_size × gradient_accumulation_steps
```

**Memory vs. Batch Size:**
- Larger batch size → More memory, better gradient estimates
- Gradient accumulation → Same effective batch size, less memory

### Loss Function Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `high_precip_threshold` | float | 10.0 | Threshold for high precipitation (mm) |
| `high_precip_weight` | float | 3.0 | Weight multiplier for high precipitation |
| `latitude_weighting` | bool | true | Apply cos(latitude) area weighting |

**Precipitation Weighting:**
- Grid points with precipitation > threshold receive higher weight
- Helps model focus on predicting significant precipitation events
- Adjust threshold based on your region's precipitation patterns

**Latitude Weighting:**
- Corrects for grid cell area variation with latitude
- Weight proportional to cos(latitude)
- Essential for fair loss computation across latitudes

### Checkpointing and Validation

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `checkpoint_frequency` | int | 1000 | Save checkpoint every N steps |
| `validation_frequency` | int | 500 | Validate every N steps |
| `early_stopping_patience` | int | 10 | Stop if no improvement for N validations |
| `save_best_only` | bool | false | Only save best model (saves disk space) |

**Checkpoint Strategy:**
- Best model saved when validation loss improves
- Periodic checkpoints saved every `checkpoint_frequency` steps
- Set `save_best_only=true` to save disk space

### Logging

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `log_level` | string | "INFO" | Logging level: "DEBUG", "INFO", "WARNING", "ERROR" |
| `log_to_file` | bool | true | Write logs to file |
| `log_to_console` | bool | true | Print logs to console |

### Device Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `device` | string | "auto" | Device: "auto", "cuda", "cpu" |
| `cuda_device` | int | 0 | GPU index if multiple GPUs |

## Creating Custom Configurations

### Step 1: Copy a Template

Start with the configuration closest to your use case:

```bash
cp convlstm/configs/baseline_12gb.yaml my_experiment.yaml
```

### Step 2: Adjust Data Paths

```yaml
data:
  data_path: "path/to/my_data.nc"
  output_dir: "checkpoints/my_experiment"
```

### Step 3: Configure Regions

Adjust region boundaries to match your study area:

```yaml
region:
  downstream:
    lat_min: 30.0  # Your target region
    lat_max: 45.0
    lon_min: 100.0
    lon_max: 120.0
```

### Step 4: Tune Hyperparameters

Adjust based on your hardware and requirements:

```yaml
training:
  batch_size: 4  # Reduce if OOM
  num_epochs: 50  # Adjust based on time budget
  learning_rate: 0.001  # Tune if needed
```

### Step 5: Run Training

```bash
python train_convlstm.py --config my_experiment.yaml
```

## Memory Optimization Guide

### If You Get OOM (Out of Memory) Errors:

1. **Reduce batch size:**
   ```yaml
   training:
     batch_size: 2  # or even 1
   ```

2. **Increase gradient accumulation:**
   ```yaml
   training:
     batch_size: 2
     gradient_accumulation_steps: 4  # Effective batch size = 8
   ```

3. **Reduce model size:**
   ```yaml
   model:
     hidden_channels: [16, 32]  # Smaller model
   ```

4. **Ensure AMP is enabled:**
   ```yaml
   training:
     use_amp: true  # Essential for memory efficiency
   ```

5. **Reduce workers:**
   ```yaml
   training:
     num_workers: 1  # Less memory overhead
   ```

## Performance Tuning Guide

### For Faster Training:

1. **Increase batch size (if memory allows):**
   ```yaml
   training:
     batch_size: 8  # Larger batches = fewer iterations
   ```

2. **Use more workers:**
   ```yaml
   training:
     num_workers: 4  # More parallel data loading
   ```

3. **Reduce validation frequency:**
   ```yaml
   checkpointing:
     validation_frequency: 1000  # Validate less often
   ```

### For Better Accuracy:

1. **Increase model capacity:**
   ```yaml
   model:
     hidden_channels: [64, 128]  # Larger model
   ```

2. **Train longer:**
   ```yaml
   training:
     num_epochs: 150  # More epochs
   ```

3. **Use larger batch size:**
   ```yaml
   training:
     batch_size: 8  # Better gradient estimates
   ```

4. **Tune loss weights:**
   ```yaml
   loss:
     high_precip_threshold: 5.0  # Lower threshold
     high_precip_weight: 5.0  # Higher weight
   ```

## Validation

After creating a custom configuration, validate it:

```bash
# Dry run to check configuration
python train_convlstm.py --config my_experiment.yaml --dry-run

# Quick test with 1 epoch
python train_convlstm.py --config my_experiment.yaml --num-epochs 1
```

## Best Practices

1. **Start with a template:** Use the configuration closest to your hardware
2. **Test quickly:** Use `quick_test.yaml` to verify your data pipeline
3. **Monitor memory:** Watch `nvidia-smi` during first epoch
4. **Save configurations:** Keep all experiment configs for reproducibility
5. **Document changes:** Add comments explaining custom settings
6. **Version control:** Track configuration files in git

## Troubleshooting

### Configuration Not Loading

**Error:** `FileNotFoundError: config file not found`

**Solution:** Use absolute path or path relative to working directory:
```bash
python train_convlstm.py --config $(pwd)/convlstm/configs/baseline_12gb.yaml
```

### Invalid Configuration

**Error:** `KeyError: missing required field`

**Solution:** Ensure all required fields are present. Compare with template files.

### Memory Issues Despite Configuration

**Error:** Still getting OOM with 8GB config

**Solution:** Your data may have higher resolution than expected. Try:
1. Check actual spatial dimensions of your data
2. Reduce `hidden_channels` further: `[8, 16]`
3. Use `batch_size: 1`

## Examples

### Example 1: Quick Sanity Check

```bash
# Test your data pipeline in <30 minutes
python train_convlstm.py --config convlstm/configs/quick_test.yaml
```

### Example 2: Standard Training

```bash
# Train baseline and upstream experiments
python train_convlstm.py --config convlstm/configs/baseline_12gb.yaml
python train_convlstm.py --config convlstm/configs/upstream_12gb.yaml
```

### Example 3: Low Memory Training

```bash
# Train on 8GB GPU
python train_convlstm.py --config convlstm/configs/baseline_8gb.yaml
```

### Example 4: High Accuracy Research

```bash
# Train high-capacity model on 24GB+ GPU
python train_convlstm.py --config convlstm/configs/high_capacity.yaml
```

## Additional Resources

- **Training Guide:** `convlstm/TRAINING_GUIDE.md`
- **Main README:** `convlstm/README.md`
- **Design Document:** `.kiro/specs/convlstm-weather-prediction/design.md`
- **Requirements:** `.kiro/specs/convlstm-weather-prediction/requirements.md`
