# Multi-Variable Evaluation Guide

This guide explains how to use the multi-variable evaluation functions for comprehensive assessment of atmospheric predictions and rolling forecasts.

## Overview

The evaluation module now supports:
1. **Per-variable metrics**: RMSE and MAE for each of the 56 output channels
2. **Per-timestep metrics**: Metrics for each step in rolling forecasts
3. **Weighted aggregated scores**: Overall metrics with configurable precipitation weighting
4. **Export to CSV/JSON**: Structured output for analysis and visualization

## Variable Structure

The 56 channels are organized as:
- **Channels 0-54**: Atmospheric variables (5 vars × 11 pressure levels)
  - DPT (Dew Point Temperature): Channels 0-10
  - GPH (Geopotential Height): Channels 11-21
  - TEM (Temperature): Channels 22-32
  - U (Eastward Wind): Channels 33-43
  - V (Northward Wind): Channels 44-54
- **Channel 55**: Precipitation

Pressure levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 100 hPa

## Usage Examples

### 1. Evaluate Multi-Variable Predictions

```python
from convlstm.evaluation import evaluate_multi_variable_predictions
import torch

# Predictions and targets: [B, 56, H, W]
predictions = torch.randn(4, 56, 50, 60)
targets = torch.randn(4, 56, 50, 60)

# Evaluate with default precipitation weight (10.0)
results = evaluate_multi_variable_predictions(
    predictions=predictions,
    targets=targets,
    precip_weight=10.0,
    output_dir="evaluation_results"
)

# Access per-variable metrics
for var_name, metrics in results['per_variable'].items():
    print(f"{var_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

# Access aggregated metrics
print(f"Weighted RMSE: {results['aggregated']['weighted_rmse']:.4f}")
print(f"Precipitation RMSE: {results['aggregated']['precip_rmse']:.4f}")
print(f"Atmospheric RMSE: {results['aggregated']['atmos_rmse']:.4f}")
```

### 2. Evaluate Rolling Forecasts

```python
from convlstm.evaluation import evaluate_rolling_forecast

# Rolling forecast predictions: [B, T, C, H, W]
predictions = torch.randn(4, 6, 56, 50, 60)  # 6 timesteps
targets = torch.randn(4, 6, 56, 50, 60)

# Evaluate rolling forecast
results = evaluate_rolling_forecast(
    predictions=predictions,
    targets=targets,
    output_dir="evaluation_results"
)

# Access per-timestep metrics
for timestep, metrics in results['per_timestep'].items():
    print(f"Timestep {timestep}: RMSE={metrics['rmse']:.4f}")

# Access per-variable metrics at each timestep (if multi-variable)
if 'per_timestep_per_variable' in results:
    timestep_1_metrics = results['per_timestep_per_variable'][1]
    print(f"Timestep 1, Precipitation: {timestep_1_metrics['precipitation']}")
```

### 3. Evaluate with Spatial Mask

```python
# Create mask for specific region (e.g., downstream region)
mask = torch.zeros(50, 60, dtype=torch.bool)
mask[10:40, 20:50] = True  # Select rectangular region

# Evaluate only on masked region
results = evaluate_multi_variable_predictions(
    predictions=predictions,
    targets=targets,
    mask=mask,
    precip_weight=10.0
)
```

### 4. Get Variable Names

```python
from convlstm.evaluation import get_variable_names

# Get list of all 56 variable names
var_names = get_variable_names()
# ['DPT_1000', 'DPT_925', ..., 'V_100', 'precipitation']
```

## Output Files

When `output_dir` is specified, the following files are created:

### Multi-Variable Evaluation
- `multi_variable_metrics.csv`: Per-variable RMSE and MAE in tabular format
- `multi_variable_metrics.json`: Complete results including aggregated metrics

### Rolling Forecast Evaluation
- `rolling_forecast_metrics.csv`: Per-timestep RMSE and MAE
- `rolling_forecast_metrics.json`: Complete results including per-variable metrics at each timestep

## Weighted Aggregation

The weighted aggregated score combines precipitation and atmospheric metrics:

```
weighted_metric = (precip_weight × precip_metric + atmos_metric) / (precip_weight + 1)
```

Default `precip_weight=10.0` means precipitation is weighted 10× more than atmospheric variables.

## Requirements Validation

This implementation validates the following requirements:

- **Requirement 10.1**: Compute RMSE and MAE for each atmospheric variable separately ✓
- **Requirement 10.2**: Compute existing metrics for precipitation predictions ✓
- **Requirement 10.3**: Report metrics for each timestep in rolling forecasts ✓
- **Requirement 10.4**: Compute weighted aggregated scores ✓
- **Requirement 10.5**: Export results to CSV/JSON with variable names and timestep labels ✓

## See Also

- `examples/evaluate_multi_variable.py`: Complete working examples
- `convlstm/evaluation.py`: Full API documentation
- Requirements document: `.kiro/specs/multi-variable-rolling-forecast/requirements.md`
