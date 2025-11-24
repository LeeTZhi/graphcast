# ConvLSTM Inference Guide

This guide explains how to use the inference module to generate precipitation predictions with a trained ConvLSTM model.

## Overview

The `convlstm/inference.py` module provides functions for:
- Loading trained models from checkpoints
- Loading normalizers for data preprocessing
- Generating single predictions
- Generating batch predictions for multiple timesteps
- Convenience functions for end-to-end inference

## Quick Start

### Basic Inference

```python
from convlstm.inference import predict_from_checkpoint
from convlstm.data import RegionConfig
import xarray as xr

# Define region configuration
region_config = RegionConfig(
    downstream_lat_min=30.0,
    downstream_lat_max=40.0,
    downstream_lon_min=110.0,
    downstream_lon_max=120.0
)

# Load test data
test_data = xr.open_dataset('data/test_data.nc')

# Generate predictions
predictions = predict_from_checkpoint(
    checkpoint_path='checkpoints/best_model.pt',
    normalizer_path='checkpoints/normalizer.pkl',
    input_data=test_data,
    region_config=region_config,
    window_size=6,
    batch_size=8
)

# Save predictions
predictions.to_netcdf('predictions/test_predictions.nc')
```

## Detailed Usage

### 1. Load Model and Normalizer Separately

```python
from convlstm.inference import load_trained_model, load_normalizer

# Load model
model, checkpoint_data = load_trained_model('checkpoints/best_model.pt')
print(f"Loaded model from epoch {checkpoint_data['epoch']}")
print(f"Best validation loss: {checkpoint_data['best_val_loss']:.4f}")

# Load normalizer
normalizer = load_normalizer('checkpoints/normalizer.pkl')
```

### 2. Generate Single Prediction

```python
from convlstm.inference import predict_single

# Prepare input window (6 timesteps)
input_window = test_data.isel(time=slice(0, 6))

# Generate prediction
prediction = predict_single(
    model=model,
    input_data=input_window,
    normalizer=normalizer,
    region_config=region_config,
    window_size=6,
    include_upstream=False
)

# Access prediction values
precip_values = prediction.precipitation.values  # Shape: (lat, lon)
```

### 3. Generate Batch Predictions

```python
from convlstm.inference import predict_batch

# Generate predictions for entire dataset
predictions = predict_batch(
    model=model,
    input_data=test_data,
    normalizer=normalizer,
    region_config=region_config,
    window_size=6,
    target_offset=1,
    include_upstream=False,
    batch_size=8
)

# Access predictions
precip_values = predictions.precipitation.values  # Shape: (time, lat, lon)
```

### 4. Inference with Upstream Region

To include upstream atmospheric data in predictions:

```python
# Define region config with upstream boundaries
region_config = RegionConfig(
    downstream_lat_min=30.0,
    downstream_lat_max=40.0,
    downstream_lon_min=110.0,
    downstream_lon_max=120.0,
    upstream_lat_min=30.0,
    upstream_lat_max=40.0,
    upstream_lon_min=100.0,
    upstream_lon_max=110.0
)

# Generate predictions with upstream data
predictions = predict_batch(
    model=model,
    input_data=test_data,
    normalizer=normalizer,
    region_config=region_config,
    window_size=6,
    include_upstream=True,  # Enable upstream
    batch_size=8
)
```

## Function Reference

### `load_trained_model(checkpoint_path, device=None)`

Loads a trained ConvLSTM model from checkpoint.

**Args:**
- `checkpoint_path`: Path to checkpoint file (.pt or .pth)
- `device`: Device to load model on (default: auto-detect)

**Returns:**
- `model`: Loaded ConvLSTMUNet model in evaluation mode
- `checkpoint_data`: Dictionary with checkpoint metadata

### `load_normalizer(normalizer_path)`

Loads a normalizer from saved file.

**Args:**
- `normalizer_path`: Path to normalizer file (.pkl)

**Returns:**
- `normalizer`: Loaded ConvLSTMNormalizer instance

### `predict_single(model, input_data, normalizer, region_config, window_size=6, include_upstream=False, device=None)`

Generates a single precipitation prediction.

**Args:**
- `model`: Trained ConvLSTMUNet model
- `input_data`: xarray Dataset with window_size timesteps
- `normalizer`: ConvLSTMNormalizer for preprocessing
- `region_config`: RegionConfig defining spatial boundaries
- `window_size`: Number of historical timesteps (default: 6)
- `include_upstream`: Whether to include upstream region (default: False)
- `device`: Device to run inference on (default: model's device)

**Returns:**
- xarray Dataset with denormalized precipitation prediction (lat, lon)

### `predict_batch(model, input_data, normalizer, region_config, window_size=6, target_offset=1, include_upstream=False, batch_size=8, device=None)`

Generates predictions for multiple timesteps using sliding windows.

**Args:**
- `model`: Trained ConvLSTMUNet model
- `input_data`: xarray Dataset with multiple timesteps
- `normalizer`: ConvLSTMNormalizer for preprocessing
- `region_config`: RegionConfig defining spatial boundaries
- `window_size`: Number of historical timesteps (default: 6)
- `target_offset`: Number of timesteps ahead to predict (default: 1)
- `include_upstream`: Whether to include upstream region (default: False)
- `batch_size`: Number of windows to process in parallel (default: 8)
- `device`: Device to run inference on (default: model's device)

**Returns:**
- xarray Dataset with denormalized precipitation predictions (time, lat, lon)

### `predict_from_checkpoint(checkpoint_path, normalizer_path, input_data, region_config, window_size=6, target_offset=1, include_upstream=False, batch_size=8, device=None)`

Convenience function to load model and generate predictions in one call.

**Args:**
- `checkpoint_path`: Path to model checkpoint file
- `normalizer_path`: Path to normalizer file
- `input_data`: xarray Dataset with atmospheric data
- `region_config`: RegionConfig defining spatial boundaries
- `window_size`: Number of historical timesteps (default: 6)
- `target_offset`: Number of timesteps ahead to predict (default: 1)
- `include_upstream`: Whether to include upstream region (default: False)
- `batch_size`: Number of windows to process in parallel (default: 8)
- `device`: Device to run inference on (default: auto-detect)

**Returns:**
- xarray Dataset with denormalized precipitation predictions

## Notes

- All predictions are automatically denormalized to original precipitation units (mm)
- Predictions are guaranteed to be non-negative (clipped at 0)
- When using upstream regions, the model processes concatenated spatial data but only outputs predictions for the downstream region
- Batch processing is more efficient than single predictions for large datasets
- GPU inference is automatically used if available

## Example: Complete Inference Workflow

```python
import xarray as xr
from convlstm.inference import predict_from_checkpoint
from convlstm.data import RegionConfig

# 1. Define region
region_config = RegionConfig(
    downstream_lat_min=30.0,
    downstream_lat_max=40.0,
    downstream_lon_min=110.0,
    downstream_lon_max=120.0
)

# 2. Load test data
test_data = xr.open_dataset('data/test_data.nc')

# 3. Generate predictions
predictions = predict_from_checkpoint(
    checkpoint_path='checkpoints/best_model.pt',
    normalizer_path='checkpoints/normalizer.pkl',
    input_data=test_data,
    region_config=region_config,
    window_size=6,
    batch_size=8
)

# 4. Save results
predictions.to_netcdf('predictions/test_predictions.nc')

# 5. Compute metrics (if ground truth available)
ground_truth = test_data.sel(
    lat=slice(region_config.downstream_lat_min, region_config.downstream_lat_max),
    lon=slice(region_config.downstream_lon_min, region_config.downstream_lon_max)
)['precipitation']

# Align time coordinates
pred_times = predictions.time.values
gt_aligned = ground_truth.sel(time=pred_times)

# Compute RMSE
rmse = ((predictions.precipitation - gt_aligned) ** 2).mean() ** 0.5
print(f"RMSE: {float(rmse.values):.4f} mm")
```
