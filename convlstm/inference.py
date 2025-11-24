"""Inference pipeline for ConvLSTM weather prediction model.

This module provides functions for loading trained models and generating
predictions on new data. It handles model loading, data preprocessing,
batch prediction, and denormalization of outputs.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple

import numpy as np
import torch
import xarray as xr

from convlstm.model import ConvLSTMUNet
from convlstm.data import ConvLSTMNormalizer, RegionConfig, stack_channels
from convlstm.trainer import load_model_checkpoint


logger = logging.getLogger(__name__)


def load_trained_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Tuple[ConvLSTMUNet, Dict[str, Any]]:
    """Load trained ConvLSTM model from checkpoint.
    
    This function loads a saved model checkpoint and reconstructs the
    ConvLSTMUNet model with the correct architecture. It returns both
    the model and checkpoint metadata for reference.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt or .pth).
        device: Device to load model on (default: auto-detect).
               If None, uses CUDA if available, otherwise CPU.
    
    Returns:
        Tuple of (model, checkpoint_data):
        - model: Loaded ConvLSTMUNet model in evaluation mode
        - checkpoint_data: Dictionary with checkpoint metadata including:
          - 'epoch': Training epoch
          - 'global_step': Training step
          - 'best_val_loss': Best validation loss
          - 'config': ConvLSTMConfig if available
          - 'region_config': RegionConfig if available
          - 'model_architecture': Model architecture details
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If checkpoint is corrupted or incompatible.
    
    Example:
        >>> model, checkpoint_data = load_trained_model('checkpoints/best_model.pt')
        >>> print(f"Loaded model from epoch {checkpoint_data['epoch']}")
        >>> print(f"Best validation loss: {checkpoint_data['best_val_loss']:.4f}")
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model from {checkpoint_path}")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint to get architecture details
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model architecture
    if 'model_architecture' not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_architecture metadata. "
            "Cannot reconstruct model."
        )
    
    arch = checkpoint['model_architecture']
    logger.info(f"Model architecture: {arch}")
    
    # Reconstruct model with correct architecture
    model = ConvLSTMUNet(
        input_channels=arch['input_channels'],
        hidden_channels=arch['hidden_channels'],
        output_channels=arch['output_channels'],
        kernel_size=arch['kernel_size']
    )
    
    # Load checkpoint using the utility function
    checkpoint_data = load_model_checkpoint(
        filepath=checkpoint_path,
        model=model,
        device=device,
        strict=True
    )
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    logger.info("Model loaded successfully and set to evaluation mode")
    logger.info(f"Checkpoint epoch: {checkpoint_data['epoch']}")
    logger.info(f"Best validation loss: {checkpoint_data['best_val_loss']:.4f}")
    
    return model, checkpoint_data


def load_normalizer(
    normalizer_path: Union[str, Path]
) -> ConvLSTMNormalizer:
    """Load normalizer from saved file.
    
    Loads a ConvLSTMNormalizer that was saved during training. The normalizer
    contains the mean and standard deviation statistics needed to normalize
    input data and denormalize predictions.
    
    Args:
        normalizer_path: Path to normalizer file (.pkl).
    
    Returns:
        Loaded ConvLSTMNormalizer instance.
    
    Raises:
        FileNotFoundError: If normalizer file doesn't exist.
    
    Example:
        >>> normalizer = load_normalizer('checkpoints/normalizer.pkl')
        >>> print(f"Normalizer variables: {list(normalizer.mean.data_vars)}")
    """
    normalizer_path = Path(normalizer_path)
    
    if not normalizer_path.exists():
        raise FileNotFoundError(f"Normalizer file not found: {normalizer_path}")
    
    logger.info(f"Loading normalizer from {normalizer_path}")
    
    normalizer = ConvLSTMNormalizer()
    normalizer.load(str(normalizer_path))
    
    logger.info("Normalizer loaded successfully")
    
    return normalizer


def predict_single(
    model: ConvLSTMUNet,
    input_data: xr.Dataset,
    normalizer: ConvLSTMNormalizer,
    region_config: RegionConfig,
    window_size: int = 6,
    include_upstream: bool = False,
    device: Optional[torch.device] = None
) -> xr.Dataset:
    """Generate a single precipitation prediction.
    
    Takes a window of historical atmospheric data and generates a precipitation
    prediction for the downstream region. Handles normalization, model inference,
    and denormalization automatically.
    
    Args:
        model: Trained ConvLSTMUNet model.
        input_data: xarray Dataset with window_size timesteps of atmospheric data.
                   Must contain all required variables (HPA variables + precipitation).
        normalizer: ConvLSTMNormalizer for preprocessing and postprocessing.
        region_config: RegionConfig defining spatial boundaries.
        window_size: Number of historical timesteps (default: 6).
        include_upstream: Whether to include upstream region in input (default: False).
        device: Device to run inference on (default: model's device).
    
    Returns:
        xarray Dataset with denormalized precipitation prediction.
        Contains a single timestep with dimensions (lat, lon).
    
    Raises:
        ValueError: If input_data has incorrect number of timesteps or missing variables.
    
    Example:
        >>> # Load model and normalizer
        >>> model, _ = load_trained_model('checkpoints/best_model.pt')
        >>> normalizer = load_normalizer('checkpoints/normalizer.pkl')
        >>> 
        >>> # Prepare input data (6 timesteps)
        >>> input_window = data.isel(time=slice(0, 6))
        >>> 
        >>> # Generate prediction
        >>> prediction = predict_single(
        ...     model=model,
        ...     input_data=input_window,
        ...     normalizer=normalizer,
        ...     region_config=region_config,
        ...     window_size=6,
        ...     include_upstream=False
        ... )
        >>> print(prediction.precipitation.values.shape)
    """
    # Validate input data
    if len(input_data.time) != window_size:
        raise ValueError(
            f"Input data must have exactly {window_size} timesteps, "
            f"got {len(input_data.time)}"
        )
    
    # Auto-detect device if not provided
    if device is None:
        device = next(model.parameters()).device
    
    # Normalize input data
    normalized_data = normalizer.normalize(input_data)
    
    # Extract regions
    downstream_data = normalized_data.sel(
        lat=slice(region_config.downstream_lat_min, region_config.downstream_lat_max),
        lon=slice(region_config.downstream_lon_min, region_config.downstream_lon_max)
    )
    
    # Prepare input tensor
    if include_upstream:
        # Extract upstream region
        upstream_data = normalized_data.sel(
            lat=slice(region_config.upstream_lat_min, region_config.upstream_lat_max),
            lon=slice(region_config.upstream_lon_min, region_config.upstream_lon_max)
        )
        
        # Stack channels for both regions
        upstream_channels = stack_channels(upstream_data, time_slice=slice(None))
        downstream_channels = stack_channels(downstream_data, time_slice=slice(None))
        
        # Concatenate along width dimension (upstream west, downstream east)
        input_array = np.concatenate(
            [upstream_channels, downstream_channels],
            axis=3  # Width dimension
        )
    else:
        # Use only downstream region
        input_array = stack_channels(downstream_data, time_slice=slice(None))
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.from_numpy(input_array).float().unsqueeze(0)  # [1, T, C, H, W]
    input_tensor = input_tensor.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)  # [1, 1, H, W]
    
    # Remove batch dimension
    prediction = prediction.squeeze(0)  # [1, H, W]
    
    # If upstream was included, extract only the downstream region from prediction
    # The model outputs the full spatial dimensions (upstream + downstream)
    # but we only want the downstream portion
    if include_upstream:
        # Calculate the width of upstream region
        upstream_width = len(upstream_data.lon)
        # Extract downstream portion (right side of concatenated output)
        prediction = prediction[:, :, upstream_width:]  # [1, H, W_down]
    
    # Get coordinates for denormalization
    lat_coords = downstream_data.lat.values
    lon_coords = downstream_data.lon.values
    
    # Denormalize prediction
    denormalized = normalizer.denormalize_tensor(
        precip_tensor=prediction,
        lat_coords=lat_coords,
        lon_coords=lon_coords
    )
    
    # Remove time dimension for single prediction (squeeze to get just lat, lon)
    if 'time' in denormalized.dims and len(denormalized.time) == 1:
        denormalized = denormalized.isel(time=0)
    
    logger.info(f"Generated prediction with shape {denormalized.precipitation.shape}")
    
    return denormalized


def predict_batch(
    model: ConvLSTMUNet,
    input_data: xr.Dataset,
    normalizer: ConvLSTMNormalizer,
    region_config: RegionConfig,
    window_size: int = 6,
    target_offset: int = 1,
    include_upstream: bool = False,
    batch_size: int = 8,
    device: Optional[torch.device] = None
) -> xr.Dataset:
    """Generate predictions for multiple timesteps using sliding windows.
    
    Processes the entire input dataset using sliding windows and generates
    predictions for all valid timesteps. This is useful for evaluating model
    performance on validation or test sets.
    
    Args:
        model: Trained ConvLSTMUNet model.
        input_data: xarray Dataset with multiple timesteps of atmospheric data.
        normalizer: ConvLSTMNormalizer for preprocessing and postprocessing.
        region_config: RegionConfig defining spatial boundaries.
        window_size: Number of historical timesteps for each prediction (default: 6).
        target_offset: Number of timesteps ahead to predict (default: 1).
        include_upstream: Whether to include upstream region in input (default: False).
        batch_size: Number of windows to process in parallel (default: 8).
        device: Device to run inference on (default: model's device).
    
    Returns:
        xarray Dataset with denormalized precipitation predictions.
        Contains predictions for all valid timesteps with dimensions (time, lat, lon).
    
    Raises:
        ValueError: If input_data has insufficient timesteps.
    
    Example:
        >>> # Load model and normalizer
        >>> model, _ = load_trained_model('checkpoints/best_model.pt')
        >>> normalizer = load_normalizer('checkpoints/normalizer.pkl')
        >>> 
        >>> # Generate predictions for entire test set
        >>> predictions = predict_batch(
        ...     model=model,
        ...     input_data=test_data,
        ...     normalizer=normalizer,
        ...     region_config=region_config,
        ...     window_size=6,
        ...     batch_size=8
        ... )
        >>> print(f"Generated {len(predictions.time)} predictions")
    """
    # Validate input data
    num_timesteps = len(input_data.time)
    min_required = window_size + target_offset
    
    if num_timesteps < min_required:
        raise ValueError(
            f"Input data has {num_timesteps} timesteps, but requires at least "
            f"{min_required} (window_size={window_size} + target_offset={target_offset})"
        )
    
    # Calculate number of valid windows
    num_windows = num_timesteps - window_size - target_offset + 1
    
    logger.info(f"Generating predictions for {num_windows} windows")
    logger.info(f"Batch size: {batch_size}")
    
    # Auto-detect device if not provided
    if device is None:
        device = next(model.parameters()).device
    
    # Normalize input data
    normalized_data = normalizer.normalize(input_data)
    
    # Extract regions
    downstream_data = normalized_data.sel(
        lat=slice(region_config.downstream_lat_min, region_config.downstream_lat_max),
        lon=slice(region_config.downstream_lon_min, region_config.downstream_lon_max)
    )
    
    if include_upstream:
        upstream_data = normalized_data.sel(
            lat=slice(region_config.upstream_lat_min, region_config.upstream_lat_max),
            lon=slice(region_config.upstream_lon_min, region_config.upstream_lon_max)
        )
    
    # Prepare to collect predictions
    all_predictions = []
    prediction_times = []
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)
            current_batch_size = batch_end - batch_start
            
            # Prepare batch of input windows
            batch_inputs = []
            
            for window_idx in range(batch_start, batch_end):
                # Calculate time indices for this window
                start_time = window_idx
                end_time = window_idx + window_size
                target_time = end_time + target_offset - 1
                
                # Store target time for this prediction
                if window_idx == batch_start:
                    # Only need to do this once per batch
                    pass
                prediction_times.append(input_data.time.values[target_time])
                
                # Create input for this window
                if include_upstream:
                    upstream_window = stack_channels(
                        upstream_data,
                        time_slice=slice(start_time, end_time)
                    )
                    downstream_window = stack_channels(
                        downstream_data,
                        time_slice=slice(start_time, end_time)
                    )
                    input_array = np.concatenate(
                        [upstream_window, downstream_window],
                        axis=3
                    )
                else:
                    input_array = stack_channels(
                        downstream_data,
                        time_slice=slice(start_time, end_time)
                    )
                
                batch_inputs.append(input_array)
            
            # Stack into batch tensor
            batch_tensor = torch.from_numpy(np.stack(batch_inputs, axis=0)).float()
            batch_tensor = batch_tensor.to(device)  # [B, T, C, H, W]
            
            # Run inference
            batch_predictions = model(batch_tensor)  # [B, 1, H, W]
            
            # If upstream was included, extract only the downstream region from predictions
            if include_upstream:
                # Calculate the width of upstream region
                upstream_width = len(upstream_data.lon)
                # Extract downstream portion (right side of concatenated output)
                batch_predictions = batch_predictions[:, :, :, upstream_width:]  # [B, 1, H, W_down]
            
            # Move to CPU and store
            batch_predictions = batch_predictions.cpu()
            all_predictions.append(batch_predictions)
            
            logger.info(f"Processed batch {batch_start // batch_size + 1}/{(num_windows + batch_size - 1) // batch_size}")
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)  # [N, 1, H, W]
    
    # Get coordinates for denormalization
    lat_coords = downstream_data.lat.values
    lon_coords = downstream_data.lon.values
    time_coords = np.array(prediction_times)
    
    # Denormalize predictions
    denormalized = normalizer.denormalize_tensor(
        precip_tensor=all_predictions,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        time_coords=time_coords
    )
    
    logger.info(f"Generated {len(denormalized.time)} predictions")
    logger.info(f"Prediction shape: {denormalized.precipitation.shape}")
    
    return denormalized


def predict_from_checkpoint(
    checkpoint_path: Union[str, Path],
    normalizer_path: Union[str, Path],
    input_data: xr.Dataset,
    region_config: RegionConfig,
    window_size: int = 6,
    target_offset: int = 1,
    include_upstream: bool = False,
    batch_size: int = 8,
    device: Optional[torch.device] = None
) -> xr.Dataset:
    """Convenience function to load model and generate predictions in one call.
    
    This is a high-level function that combines model loading and batch prediction.
    It's useful for quick inference without manually managing model and normalizer
    loading.
    
    Args:
        checkpoint_path: Path to model checkpoint file.
        normalizer_path: Path to normalizer file.
        input_data: xarray Dataset with atmospheric data.
        region_config: RegionConfig defining spatial boundaries.
        window_size: Number of historical timesteps (default: 6).
        target_offset: Number of timesteps ahead to predict (default: 1).
        include_upstream: Whether to include upstream region (default: False).
        batch_size: Number of windows to process in parallel (default: 8).
        device: Device to run inference on (default: auto-detect).
    
    Returns:
        xarray Dataset with denormalized precipitation predictions.
    
    Example:
        >>> predictions = predict_from_checkpoint(
        ...     checkpoint_path='checkpoints/best_model.pt',
        ...     normalizer_path='checkpoints/normalizer.pkl',
        ...     input_data=test_data,
        ...     region_config=region_config
        ... )
    """
    # Load model and normalizer
    model, checkpoint_data = load_trained_model(checkpoint_path, device=device)
    normalizer = load_normalizer(normalizer_path)
    
    # Log checkpoint information
    logger.info(f"Using model from epoch {checkpoint_data['epoch']}")
    logger.info(f"Best validation loss: {checkpoint_data['best_val_loss']:.4f}")
    
    # Generate predictions
    predictions = predict_batch(
        model=model,
        input_data=input_data,
        normalizer=normalizer,
        region_config=region_config,
        window_size=window_size,
        target_offset=target_offset,
        include_upstream=include_upstream,
        batch_size=batch_size,
        device=device
    )
    
    return predictions
