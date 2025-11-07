# Regional Weather Prediction Scripts

This directory contains executable scripts for the complete Regional Weather Prediction workflow.

## Scripts Overview

### 1. `preprocess_data.py`
Converts raw text files into structured NetCDF format.

**Usage:**
```bash
python scripts/preprocess_data.py \
    --lat-file data/raw/Lat.txt \
    --lon-file data/raw/Lon.txt \
    --hpa-dir data/raw/HPA \
    --precip-dir data/raw/precipitation \
    --output data/processed/regional_weather.nc
```

**Key Options:**
- `--compression`: NetCDF compression level (0-9, default: 4)
- `--verbose`: Enable debug logging

### 2. `train_model.py`
Trains the Regional GNN model on preprocessed data with GPU optimizations.

**Basic Usage:**
```bash
python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir checkpoints/experiment_1 \
    --latent-size 256 \
    --num-gnn-layers 12 \
    --learning-rate 1e-4 \
    --num-epochs 100
```

**GPU-Optimized Usage (Recommended):**
```bash
python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir checkpoints/gpu_optimized \
    --batch-size 8 \
    --use-prefetch \
    --prefetch-buffer-size 8 \
    --learning-rate 1e-4 \
    --num-epochs 100
```

**Resume Training from Checkpoint:**
```bash
python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir checkpoints/gpu_optimized \
    --resume-from checkpoints/gpu_optimized/checkpoint_step_5000.pkl \
    --num-epochs 100
```

**Key Options:**
- `--latent-size`: Latent dimension (default: 256)
- `--num-gnn-layers`: Number of GNN layers (default: 12)
- `--learning-rate`: Initial learning rate (default: 1e-4)
- `--batch-size`: Training batch size (default: 4)
- `--num-epochs`: Number of training epochs (default: 100)
- `--train-end-year`: Last year for training (default: 2018)
- `--val-end-year`: Last year for validation (default: 2019)
- `--resume-from`: Path to checkpoint to resume training from

**GPU Optimization Options:**
- `--use-prefetch`: Enable data prefetching (default: True)
- `--no-prefetch`: Disable data prefetching
- `--prefetch-buffer-size`: Prefetch buffer size (default: 4)
- `--jax-platform`: Force JAX platform (cpu/gpu/tpu)

**Performance Tips:**
- Use `--batch-size 8` or higher for better GPU utilization
- Enable prefetching with `--use-prefetch` (default)
- First training step will be slow (JIT compilation), subsequent steps are fast
- See `docs/GPU_optimization_guide.md` for detailed optimization guide

### 3. `run_inference.py`
Generates precipitation predictions using a trained model.

**Single-step prediction:**
```bash
python scripts/run_inference.py \
    --data data/processed/regional_weather.nc \
    --checkpoint checkpoints/experiment_1/best_model.pkl \
    --normalizer checkpoints/experiment_1/normalizer.pkl \
    --target-time "2020-01-15 12:00:00" \
    --output predictions/pred_20200115_12.nc
```

**Multi-step prediction:**
```bash
python scripts/run_inference.py \
    --data data/processed/regional_weather.nc \
    --checkpoint checkpoints/experiment_1/best_model.pkl \
    --normalizer checkpoints/experiment_1/normalizer.pkl \
    --initial-time "2020-01-15 00:00:00" \
    --num-steps 10 \
    --output predictions/pred_sequence.nc
```

**Key Options:**
- `--target-time`: Target timestamp for single-step prediction
- `--initial-time`: Initial timestamp for multi-step prediction
- `--num-steps`: Number of 12-hour steps (for multi-step mode)

### 4. `evaluate_model.py`
Evaluates model performance on test set with metrics and visualizations.

**Usage:**
```bash
python scripts/evaluate_model.py \
    --data data/processed/regional_weather.nc \
    --checkpoint checkpoints/experiment_1/best_model.pkl \
    --normalizer checkpoints/experiment_1/normalizer.pkl \
    --test-start-time "2020-01-01 00:00:00" \
    --test-end-time "2020-12-31 23:59:59" \
    --output-dir evaluation/experiment_1 \
    --visualizations
```

**Key Options:**
- `--test-start-time`: Start of test period (required)
- `--test-end-time`: End of test period (optional)
- `--visualizations`: Create visualization plots
- `--verbose`: Enable debug logging

## Complete Workflow Example

```bash
# Step 1: Preprocess data
python scripts/preprocess_data.py \
    --lat-file data/raw/Lat.txt \
    --lon-file data/raw/Lon.txt \
    --hpa-dir data/raw/HPA \
    --precip-dir data/raw/precipitation \
    --output data/processed/regional_weather.nc

# Step 2: Train model
python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir checkpoints/experiment_1 \
    --num-epochs 100

# Step 3: Generate predictions
python scripts/run_inference.py \
    --data data/processed/regional_weather.nc \
    --checkpoint checkpoints/experiment_1/best_model.pkl \
    --normalizer checkpoints/experiment_1/normalizer.pkl \
    --target-time "2020-01-15 12:00:00" \
    --output predictions/pred_20200115_12.nc

# Step 4: Evaluate model
python scripts/evaluate_model.py \
    --data data/processed/regional_weather.nc \
    --checkpoint checkpoints/experiment_1/best_model.pkl \
    --normalizer checkpoints/experiment_1/normalizer.pkl \
    --test-start-time "2020-01-01 00:00:00" \
    --output-dir evaluation/experiment_1 \
    --visualizations
```

## Configuration

All scripts support configuration of region boundaries and model architecture through command-line arguments. For consistent results, use the same configuration across training, inference, and evaluation.

### Region Configuration
- `--downstream-lat-min/max`: Downstream region latitude bounds
- `--downstream-lon-min/max`: Downstream region longitude bounds
- `--upstream-lat-min/max`: Upstream region latitude bounds
- `--upstream-lon-min/max`: Upstream region longitude bounds

### Model Configuration
- `--latent-size`: Latent dimension
- `--num-gnn-layers`: Number of GNN layers
- `--mlp-hidden-size`: MLP hidden layer size
- `--mlp-num-hidden-layers`: Number of MLP hidden layers

## Output Files

### Training Output
- `best_model.pkl`: Best model checkpoint with full training state (params, optimizer state, epoch, step, best_val_loss)
- `normalizer.pkl`: Data normalization statistics
- `checkpoint_step_*.pkl`: Periodic checkpoints with full training state for resuming
- `training.log`: Training logs

### Inference Output
- `*.nc`: NetCDF file with precipitation predictions

### Evaluation Output
- `metrics.json`: Evaluation metrics (JSON format)
- `metrics.txt`: Evaluation metrics (human-readable)
- `predictions.nc`: All predictions on test set
- `evaluation.log`: Evaluation logs
- `visualizations/`: Comparison plots and time series (if --visualizations enabled)

## GPU Optimization

The training script includes several GPU optimizations for faster training:

1. **JIT Compilation**: Training steps are compiled with JAX for 5-10x speedup
2. **Data Prefetching**: Background data loading keeps GPU busy (20-40% faster)
3. **Automatic Device Placement**: JIT functions automatically use GPU
4. **Efficient Memory**: Minimizes CPU-GPU data movement

**Check GPU Setup:**
```bash
python scripts/check_gpu.py
```

**Expected Performance:**
- Baseline: ~2.0s per step, 40-60% GPU utilization
- Optimized: ~0.15s per step, 90-98% GPU utilization
- Overall speedup: ~13x on NVIDIA A100

See `docs/GPU_optimization_guide.md` for comprehensive optimization guide.

## Tips

1. **Start Small**: Test with a small subset of data first
2. **Monitor Training**: Check `training.log` for convergence
3. **GPU Usage**: Run `python scripts/check_gpu.py` to verify GPU is available
4. **Batch Size**: Increase batch size (8, 16, 32) for better GPU utilization
5. **Checkpoints**: Save checkpoints frequently to avoid losing progress
6. **Visualizations**: Always create visualizations to inspect prediction quality
7. **First Step Slow**: First training step takes 30s-2min for JIT compilation (normal)

## Troubleshooting

See the main [README](../graphcast_regional/README.md#troubleshooting) for common issues and solutions.
