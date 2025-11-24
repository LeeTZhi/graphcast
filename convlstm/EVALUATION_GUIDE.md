# ConvLSTM Evaluation Guide

This guide explains how to use the evaluation metrics module to assess precipitation prediction accuracy.

## Overview

The `convlstm/evaluation.py` module provides comprehensive evaluation metrics for comparing precipitation predictions against ground truth. It supports:

- **RMSE (Root Mean Squared Error)**: Measures average prediction error magnitude
- **MAE (Mean Absolute Error)**: Measures average absolute prediction error
- **CSI (Critical Success Index)**: Measures skill in predicting precipitation events above thresholds

All metrics can be computed specifically for the downstream region to ensure fair comparison between experiments.

## Basic Usage

### Computing Individual Metrics

```python
import torch
from convlstm.evaluation import compute_rmse, compute_mae, compute_csi

# Predictions and targets as PyTorch tensors [B, H, W]
predictions = torch.rand(4, 50, 60) * 20  # 4 samples, 50x60 grid, 0-20mm
targets = torch.rand(4, 50, 60) * 20

# Compute RMSE
rmse = compute_rmse(predictions, targets)
print(f"RMSE: {rmse:.4f} mm")

# Compute MAE
mae = compute_mae(predictions, targets)
print(f"MAE: {mae:.4f} mm")

# Compute CSI at 10mm threshold
csi = compute_csi(predictions, targets, threshold=10.0)
print(f"CSI @ 10mm: {csi:.4f}")
```

### Computing Metrics for Downstream Region Only

```python
import numpy as np
from convlstm.evaluation import create_downstream_mask, compute_metrics_for_region

# Create coordinate arrays
lats = np.linspace(20, 50, 50)
lons = np.linspace(70, 130, 80)

# Create downstream mask (25-40°N, 110-125°E)
downstream_mask = create_downstream_mask(
    lats, lons,
    downstream_lat_min=25,
    downstream_lat_max=40,
    downstream_lon_min=110,
    downstream_lon_max=125
)

# Compute all metrics for downstream region
metrics = compute_metrics_for_region(
    predictions,
    targets,
    mask=downstream_mask,
    csi_thresholds=[1.0, 5.0, 10.0, 25.0]
)

print(f"RMSE: {metrics['rmse']:.4f} mm")
print(f"MAE: {metrics['mae']:.4f} mm")
print(f"CSI @ 1mm: {metrics['csi_1mm']:.4f}")
print(f"CSI @ 10mm: {metrics['csi_10mm']:.4f}")
```

## Comparing Two Experiments

The primary use case is comparing the baseline experiment (downstream only) with the upstream experiment (downstream + upstream):

```python
from convlstm.evaluation import compare_experiments

# Predictions from both experiments [B, H, W]
baseline_predictions = torch.rand(10, 50, 60) * 20
upstream_predictions = torch.rand(10, 50, 60) * 20
targets = torch.rand(10, 50, 60) * 20

# Compare on downstream region
exp1_metrics, exp2_metrics = compare_experiments(
    baseline_predictions,
    upstream_predictions,
    targets,
    downstream_mask,
    exp1_name="Baseline (Downstream Only)",
    exp2_name="Upstream (Downstream + Upstream)",
    csi_thresholds=[1.0, 5.0, 10.0, 25.0]
)

# Results are printed automatically with improvement percentages
# You can also access metrics programmatically:
print(f"\nBaseline RMSE: {exp1_metrics['rmse']:.4f} mm")
print(f"Upstream RMSE: {exp2_metrics['rmse']:.4f} mm")

improvement = (exp1_metrics['rmse'] - exp2_metrics['rmse']) / exp1_metrics['rmse'] * 100
print(f"RMSE Improvement: {improvement:+.2f}%")
```

## Working with xarray Datasets

If you have predictions and targets as xarray Datasets (e.g., from model inference):

```python
import xarray as xr
from convlstm.evaluation import compute_metrics_from_xarray

# Load predictions and targets
predictions_ds = xr.open_dataset('predictions.nc')
targets_ds = xr.open_dataset('targets.nc')

# Compute metrics for downstream region
metrics = compute_metrics_from_xarray(
    predictions_ds,
    targets_ds,
    downstream_lat_min=25,
    downstream_lat_max=40,
    downstream_lon_min=110,
    downstream_lon_max=125,
    csi_thresholds=[1.0, 5.0, 10.0, 25.0]
)

print(f"RMSE: {metrics['rmse']:.4f} mm")
print(f"MAE: {metrics['mae']:.4f} mm")
```

## Integration with Training/Inference

### During Training

```python
from convlstm.trainer import ConvLSTMTrainer
from convlstm.evaluation import compute_metrics_for_region, create_downstream_mask

# After validation epoch
val_predictions = []  # Collect predictions during validation
val_targets = []      # Collect targets during validation

# Stack into tensors
predictions_tensor = torch.stack(val_predictions)
targets_tensor = torch.stack(val_targets)

# Compute metrics
metrics = compute_metrics_for_region(
    predictions_tensor,
    targets_tensor,
    mask=downstream_mask
)

print(f"Validation RMSE: {metrics['rmse']:.4f} mm")
```

### After Inference

```python
from convlstm.inference import load_model, predict
from convlstm.evaluation import compute_metrics_from_xarray

# Load model and make predictions
model, normalizer, config = load_model('checkpoints/best_model.pkl')
predictions_ds = predict(model, normalizer, test_data, config)

# Load ground truth
targets_ds = xr.open_dataset('test_targets.nc')

# Evaluate
metrics = compute_metrics_from_xarray(
    predictions_ds,
    targets_ds,
    downstream_lat_min=config.downstream_lat_min,
    downstream_lat_max=config.downstream_lat_max,
    downstream_lon_min=config.downstream_lon_min,
    downstream_lon_max=config.downstream_lon_max
)
```

## Understanding the Metrics

### RMSE (Root Mean Squared Error)
- **Range**: [0, ∞), lower is better
- **Units**: Same as predictions (mm)
- **Interpretation**: Average magnitude of errors, sensitive to large errors
- **Use case**: Overall prediction accuracy assessment

### MAE (Mean Absolute Error)
- **Range**: [0, ∞), lower is better
- **Units**: Same as predictions (mm)
- **Interpretation**: Average absolute error, less sensitive to outliers than RMSE
- **Use case**: Robust measure of typical prediction error

### CSI (Critical Success Index)
- **Range**: [0, 1], higher is better
- **Units**: Dimensionless
- **Interpretation**: Fraction of correctly predicted events relative to all predicted and observed events
- **Formula**: CSI = hits / (hits + misses + false_alarms)
- **Use case**: Skill in predicting precipitation events above specific thresholds

**CSI Thresholds:**
- **1mm**: Light precipitation events (most common)
- **5mm**: Moderate precipitation events
- **10mm**: Heavy precipitation events
- **25mm**: Very heavy precipitation events (rare but important)

## Best Practices

1. **Always use downstream mask for comparison**: When comparing experiments, ensure both use the same downstream region mask for fair comparison.

2. **Use multiple thresholds for CSI**: Different precipitation intensities have different prediction challenges. Use multiple thresholds to get a complete picture.

3. **Consider both RMSE and MAE**: RMSE is more sensitive to large errors, while MAE is more robust. Both provide complementary information.

4. **Check for NaN in CSI**: CSI returns NaN when no events occur at a threshold. This is expected for high thresholds with limited data.

5. **Normalize by region size**: The metrics automatically handle different region sizes by computing means, so results are comparable across different spatial extents.

## Example: Complete Evaluation Workflow

```python
import torch
import numpy as np
from convlstm.evaluation import (
    create_downstream_mask,
    compare_experiments
)

# Setup
lats = np.linspace(20, 50, 100)
lons = np.linspace(70, 130, 150)

downstream_mask = create_downstream_mask(
    lats, lons, 25, 40, 110, 125
)

# Load predictions from both experiments
baseline_pred = torch.load('baseline_predictions.pt')
upstream_pred = torch.load('upstream_predictions.pt')
targets = torch.load('targets.pt')

# Compare experiments
exp1_metrics, exp2_metrics = compare_experiments(
    baseline_pred,
    upstream_pred,
    targets,
    downstream_mask,
    exp1_name="Baseline (Downstream Only)",
    exp2_name="Upstream (Downstream + Upstream)"
)

# Save results
import json
with open('evaluation_results.json', 'w') as f:
    json.dump({
        'baseline': exp1_metrics,
        'upstream': exp2_metrics
    }, f, indent=2)
```

## Troubleshooting

### Shape Mismatch Errors
Ensure predictions and targets have compatible shapes:
- Predictions: `[B, H, W]` or `[B, 1, H, W]`
- Targets: `[B, H, W]`
- Mask: `[H, W]` (will be expanded automatically)

### NaN in CSI
This is expected when no precipitation events occur above the threshold. Consider:
- Using lower thresholds
- Checking if your data has sufficient precipitation events
- Verifying the threshold is appropriate for your region

### Memory Issues with Large Datasets
For very large datasets, process in batches:
```python
batch_size = 100
all_rmse = []

for i in range(0, len(predictions), batch_size):
    batch_pred = predictions[i:i+batch_size]
    batch_target = targets[i:i+batch_size]
    rmse = compute_rmse(batch_pred, batch_target, mask)
    all_rmse.append(rmse)

# Weighted average by batch size
final_rmse = np.mean(all_rmse)
```
