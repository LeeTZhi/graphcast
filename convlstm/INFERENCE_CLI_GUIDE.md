# ConvLSTM Inference CLI Guide

This guide explains how to use the `run_inference_convlstm.py` script for generating predictions with trained ConvLSTM models.

## Quick Start

### Basic Inference

Generate predictions on test data:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --normalizer checkpoints/baseline/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/baseline
```

This will:
- Load the trained model and normalizer
- Generate predictions for all valid timesteps in the data
- Save predictions to `outputs/baseline/predictions.nc`

### Inference with Visualizations

Generate predictions and create visualization plots:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --normalizer checkpoints/baseline/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/baseline \
    --visualize \
    --viz-timesteps 0 5 10 15 20
```

This will create:
- Precipitation prediction maps
- Error maps (prediction - ground truth)
- All visualizations saved to `outputs/baseline/visualizations/`

### Inference with Upstream Region

If your model was trained with upstream data:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/with_upstream/best_model.pt \
    --normalizer checkpoints/with_upstream/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/with_upstream \
    --include-upstream
```

**Important**: The `--include-upstream` flag must match your training configuration!

### Comparing Two Experiments

Compare baseline vs. upstream models side-by-side:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --normalizer checkpoints/baseline/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/comparison \
    --compare-checkpoint checkpoints/with_upstream/best_model.pt \
    --compare-normalizer checkpoints/with_upstream/normalizer.pkl \
    --compare-upstream \
    --exp1-name "Baseline (Downstream Only)" \
    --exp2-name "With Upstream Data" \
    --visualize
```

This will create:
- Side-by-side comparison plots (ground truth, exp1, exp2)
- Error comparison maps for both experiments
- All saved to `outputs/comparison/visualizations/`

## Command Line Arguments

### Required Arguments

- `--checkpoint`: Path to model checkpoint file (`.pt` or `.pth`)
- `--normalizer`: Path to normalizer file (`.pkl`)
- `--data`: Path to input NetCDF data file
- `--output-dir`: Directory to save predictions and visualizations

### Region Configuration

Configure spatial boundaries (must match training configuration):

- `--include-upstream`: Include upstream region in input
- `--downstream-lat-min`: Downstream minimum latitude (default: 25.0)
- `--downstream-lat-max`: Downstream maximum latitude (default: 40.0)
- `--downstream-lon-min`: Downstream minimum longitude (default: 110.0)
- `--downstream-lon-max`: Downstream maximum longitude (default: 125.0)
- `--upstream-lat-min`: Upstream minimum latitude (default: 25.0)
- `--upstream-lat-max`: Upstream maximum latitude (default: 50.0)
- `--upstream-lon-min`: Upstream minimum longitude (default: 70.0)
- `--upstream-lon-max`: Upstream maximum longitude (default: 110.0)

### Inference Configuration

- `--window-size`: Number of historical timesteps (default: 6)
- `--target-offset`: Timesteps ahead to predict (default: 1)
- `--batch-size`: Batch size for inference (default: 8)

### Comparison Experiment

- `--compare-checkpoint`: Second model checkpoint for comparison
- `--compare-normalizer`: Second normalizer for comparison
- `--compare-upstream`: Second model uses upstream region
- `--exp1-name`: Name for first experiment (default: "Experiment 1")
- `--exp2-name`: Name for second experiment (default: "Experiment 2")

### Visualization Options

- `--visualize`: Generate visualization plots
- `--viz-timesteps`: Specific timestep indices to visualize (default: first 5)
- `--viz-format`: Format for saved visualizations: png, pdf, jpg, svg (default: png)
- `--viz-dpi`: DPI for saved visualizations (default: 150)

### Output Options

- `--save-predictions`: Save predictions to NetCDF (default: True)
- `--no-save-predictions`: Don't save predictions to NetCDF
- `--predictions-filename`: Filename for predictions (default: predictions.nc)

### Device and Logging

- `--device`: Device to use: auto, cuda, cpu (default: auto)
- `--log-level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)

## Output Files

### Predictions

When `--save-predictions` is enabled (default):

- `predictions.nc`: NetCDF file with all predictions
  - Dimensions: (time, lat, lon)
  - Variable: precipitation (mm)
  - Coordinates: time, lat, lon

If comparing two experiments:
- `comparison_predictions.nc`: Predictions from second model

### Visualizations

When `--visualize` is enabled, creates `visualizations/` directory with:

**Single Experiment:**
- `{exp_name}_prediction_{timestep}.png`: Precipitation prediction map
- `{exp_name}_error_{timestep}.png`: Error map (prediction - truth)

**Comparison Mode:**
- `comparison_comparison_{timestep}.png`: Side-by-side comparison (truth, exp1, exp2)
- `error_comparison_multi_error_{timestep}.png`: Side-by-side error maps

### Logs

- `inference.log`: Detailed log of inference process

## Examples

### Example 1: Quick Evaluation

Evaluate model on test set and save predictions:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/best_model.pt \
    --normalizer checkpoints/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/evaluation
```

### Example 2: Visual Inspection

Generate visualizations for specific timesteps:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/best_model.pt \
    --normalizer checkpoints/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/visual_inspection \
    --visualize \
    --viz-timesteps 0 10 20 30 40 \
    --viz-format pdf \
    --viz-dpi 300
```

### Example 3: Full Comparison Study

Compare baseline and upstream models with high-quality visualizations:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --normalizer checkpoints/baseline/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/full_comparison \
    --compare-checkpoint checkpoints/upstream/best_model.pt \
    --compare-normalizer checkpoints/upstream/normalizer.pkl \
    --compare-upstream \
    --exp1-name "Baseline" \
    --exp2-name "With_Upstream" \
    --visualize \
    --viz-format png \
    --viz-dpi 200
```

### Example 4: CPU-Only Inference

Run inference on CPU (useful for machines without GPU):

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/best_model.pt \
    --normalizer checkpoints/normalizer.pkl \
    --data data/test_data.nc \
    --output-dir outputs/cpu_inference \
    --device cpu \
    --batch-size 4
```

## Tips and Best Practices

1. **Match Training Configuration**: Ensure `--include-upstream` and region boundaries match your training setup

2. **Batch Size**: Adjust `--batch-size` based on available memory:
   - GPU with 12GB: batch_size=8-16
   - GPU with 8GB: batch_size=4-8
   - CPU: batch_size=1-4

3. **Visualization Selection**: For large datasets, specify `--viz-timesteps` to avoid generating too many plots

4. **Output Organization**: Use descriptive `--output-dir` names:
   - `outputs/baseline_eval`
   - `outputs/upstream_eval`
   - `outputs/baseline_vs_upstream`

5. **Comparison Studies**: When comparing experiments, use meaningful names with `--exp1-name` and `--exp2-name`

6. **High-Quality Figures**: For publications, use:
   - `--viz-format pdf`
   - `--viz-dpi 300`

## Troubleshooting

### Out of Memory

If you encounter CUDA out of memory errors:

```bash
# Reduce batch size
python run_inference_convlstm.py ... --batch-size 4

# Or use CPU
python run_inference_convlstm.py ... --device cpu
```

### Missing Ground Truth

If visualizations fail due to missing ground truth:
- Ensure your data file contains the timesteps being predicted
- Check that prediction times align with data times
- Use `--log-level DEBUG` for detailed information

### Region Mismatch

If you get errors about region boundaries:
- Verify region arguments match your training configuration
- Check that data covers the specified regions
- Use the same region config as training

## See Also

- [Training Guide](TRAINING_GUIDE.md): How to train ConvLSTM models
- [Inference Guide](INFERENCE_GUIDE.md): Programmatic inference API
- [README](README.md): Overview of ConvLSTM module
