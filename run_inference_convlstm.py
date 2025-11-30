#!/usr/bin/env python3
"""Inference script for ConvLSTM weather prediction model.

This script provides a comprehensive CLI for running inference with trained
ConvLSTM models. It supports:
- Loading trained models from checkpoints
- Generating predictions on new data
- Saving predictions to NetCDF format
- Creating visualizations (precipitation maps, error maps, comparisons)
- Batch processing of multiple timesteps

Example usage:
    # Basic inference - generate predictions
    python run_inference_convlstm.py \\
        --checkpoint checkpoints/baseline/best_model.pt \\
        --normalizer checkpoints/baseline/normalizer.pkl \\
        --data data/test_data.nc \\
        --output-dir outputs/baseline

    # Inference with visualizations
    python run_inference_convlstm.py \\
        --checkpoint checkpoints/baseline/best_model.pt \\
        --normalizer checkpoints/baseline/normalizer.pkl \\
        --data data/test_data.nc \\
        --output-dir outputs/baseline \\
        --visualize \\
        --viz-timesteps 0 5 10

    # Inference with upstream region
    python run_inference_convlstm.py \\
        --checkpoint checkpoints/with_upstream/best_model.pt \\
        --normalizer checkpoints/with_upstream/normalizer.pkl \\
        --data data/test_data.nc \\
        --output-dir outputs/with_upstream \\
        --include-upstream

    # Inference for specific time range
    python run_inference_convlstm.py \\
        --checkpoint checkpoints/baseline/best_model.pt \\
        --normalizer checkpoints/baseline/normalizer.pkl \\
        --data data/test_data.nc \\
        --output-dir outputs/baseline \\
        --start-time 2023-01-01T00:00:00 \\
        --end-time 2023-01-31T23:00:00

    # Inference for specific times
    python run_inference_convlstm.py \\
        --checkpoint checkpoints/baseline/best_model.pt \\
        --normalizer checkpoints/baseline/normalizer.pkl \\
        --data data/test_data.nc \\
        --output-dir outputs/baseline \\
        --specific-times 2023-01-15T12:00:00 2023-02-15T12:00:00

    # Compare two experiments
    python run_inference_convlstm.py \\
        --checkpoint checkpoints/baseline/best_model.pt \\
        --normalizer checkpoints/baseline/normalizer.pkl \\
        --data data/test_data.nc \\
        --output-dir outputs/comparison \\
        --compare-checkpoint checkpoints/with_upstream/best_model.pt \\
        --compare-normalizer checkpoints/with_upstream/normalizer.pkl \\
        --compare-upstream \\
        --visualize
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import torch
import xarray as xr

from convlstm.model import ConvLSTMUNet
from convlstm.model_deep import DeepConvLSTMUNet
from convlstm.model_dual_stream import DualStreamConvLSTMUNet
from convlstm.model_dual_stream_deep import DeepDualStreamConvLSTMUNet
from convlstm.trainer import load_model_checkpoint
from convlstm.inference import (
    load_normalizer,
    predict_batch,
    predict_single
)
from convlstm.data import RegionConfig
from convlstm.visualization import (
    plot_precipitation_map,
    create_comparison_plot,
    create_error_map,
    save_plot,
    create_multi_error_comparison
)
import numpy as np


def load_trained_model_auto(checkpoint_path: str, device: torch.device, model_type_override: str = None):
    """Load trained model with automatic architecture detection.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        model_type_override: Optional model type to override auto-detection
                            (useful for old checkpoints without metadata)
        
    Returns:
        Tuple of (model, checkpoint_data, model_type)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint to inspect architecture
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_architecture' not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain model_architecture metadata. "
            "Cannot reconstruct model."
        )
    
    arch = checkpoint['model_architecture']
    hidden_channels = arch['hidden_channels']
    
    # Check if checkpoint has regularization layers by inspecting state_dict keys
    state_dict = checkpoint['model_state_dict']
    has_batch_norm = any('_bn' in key or 'batch_norm' in key for key in state_dict.keys())
    has_dropout_layers = any('dropout' in key for key in state_dict.keys())
    
    # Check if it's a dual-stream model by looking for upstream encoder layers
    is_dual_stream = any('upstream_encoder' in key for key in state_dict.keys())
    
    # Get dropout rate from architecture or infer from state dict
    dropout_rate = arch.get('dropout_rate', 0.0)
    if dropout_rate == 0.0 and has_dropout_layers:
        dropout_rate = 0.2  # Default if dropout layers exist but rate not saved
    
    # Get model type from architecture metadata (if available)
    model_type = arch.get('model_type', None)
    
    print(f"Checkpoint analysis:")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Has batch norm layers: {has_batch_norm}")
    print(f"  Has dropout layers: {has_dropout_layers}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Is dual-stream: {is_dual_stream}")
    print(f"  Model type (from metadata): {model_type}")
    
    # Use override if provided
    if model_type_override is not None:
        print(f"  Model type (override): {model_type_override}")
        model_type = model_type_override
    
    # Detect model type based on architecture
    if model_type is None:
        # Fallback: infer from structure
        if is_dual_stream:
            if len(hidden_channels) == 2:
                model_type = 'dual_stream'
            else:
                model_type = 'dual_stream_deep'
        else:
            if len(hidden_channels) == 2:
                model_type = 'shallow'
            else:
                model_type = 'deep'
    
    print(f"  Detected model type: {model_type}")
    
    # Create model based on detected type
    if model_type == 'shallow':
        model = ConvLSTMUNet(
            input_channels=arch['input_channels'],
            hidden_channels=arch['hidden_channels'],
            output_channels=arch['output_channels'],
            kernel_size=arch['kernel_size'],
            use_attention=arch.get('use_attention', True),
            use_group_norm=arch.get('use_group_norm', True)
        )
    elif model_type == 'deep':
        model = DeepConvLSTMUNet(
            input_channels=arch['input_channels'],
            hidden_channels=arch['hidden_channels'],
            output_channels=arch['output_channels'],
            kernel_size=arch['kernel_size'],
            dropout_rate=dropout_rate,
            use_batch_norm=has_batch_norm,
            use_group_norm=arch.get('use_group_norm', True),
            use_spatial_dropout=arch.get('use_spatial_dropout', True),
            use_attention=arch.get('use_attention', True)
        )
        print(f"  Created DeepConvLSTMUNet with:")
        print(f"    dropout_rate={dropout_rate}")
        print(f"    use_batch_norm={has_batch_norm}")
        print(f"    use_spatial_dropout={arch.get('use_spatial_dropout', True)}")
    elif model_type == 'dual_stream':
        model = DualStreamConvLSTMUNet(
            input_channels=arch['input_channels'],
            hidden_channels=arch['hidden_channels'],
            output_channels=arch['output_channels'],
            kernel_size=arch['kernel_size'],
            use_attention=arch.get('use_attention', True),
            use_group_norm=arch.get('use_group_norm', True),
            dropout_rate=dropout_rate
        )
        print(f"  Created DualStreamConvLSTMUNet with:")
        print(f"    dropout_rate={dropout_rate}")
    elif model_type == 'dual_stream_deep':
        model = DeepDualStreamConvLSTMUNet(
            input_channels=arch['input_channels'],
            hidden_channels=arch['hidden_channels'],
            output_channels=arch['output_channels'],
            kernel_size=arch['kernel_size'],
            use_attention=arch.get('use_attention', True),
            use_group_norm=arch.get('use_group_norm', True),
            dropout_rate=dropout_rate
        )
        print(f"  Created DeepDualStreamConvLSTMUNet with:")
        print(f"    dropout_rate={dropout_rate}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint (use strict=False for backward compatibility)
    checkpoint_data = load_model_checkpoint(
        filepath=checkpoint_path,
        model=model,
        device=device,
        strict=False  # Allow loading old checkpoints without new layers
    )
    
    # Ensure model is on the correct device (critical for CUDA)
    model = model.to(device)
    
    # Verify all parameters are on correct device
    param_devices = set()
    for name, param in model.named_parameters():
        param_devices.add(str(param.device))
    
    if len(param_devices) > 1:
        print(f"WARNING: Model parameters are on multiple devices: {param_devices}")
        print("Forcing all parameters to target device...")
        model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    # Final verification
    model_device = next(model.parameters()).device
    print(f"Model loaded on device: {model_device}")
    
    # Check device compatibility (handle mps vs mps:0 difference)
    device_type_match = model_device.type == device.type
    if not device_type_match:
        print(f"WARNING: Model device type ({model_device.type}) != target device type ({device.type})")
        print("Attempting to fix...")
        model = model.to(device)
        model_device = next(model.parameters()).device
        print(f"Model now on device: {model_device}")
    else:
        print(f"✓ Model device type matches target: {device.type}")
    
    return model, checkpoint_data, model_type


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        output_dir: Directory to save log file
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = output_dir / "inference.log"
    
    # Create logger
    logger = logging.getLogger("run_inference_convlstm")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def validate_date(date_str: str) -> bool:
    """Validate if a date string is valid.
    
    Args:
        date_str: Date string in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        
    Returns:
        True if valid, False otherwise
    """
    from datetime import datetime
    
    try:
        # Try parsing with time
        if 'T' in date_str:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            # Try parsing date only
            datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def load_data(
    data_path: str, 
    logger: logging.Logger,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    specific_times: Optional[List[str]] = None,
    window_size: Optional[int] = None,
    target_offset: Optional[int] = None
) -> xr.Dataset:
    """Load xarray dataset from file with optional time filtering.
    
    When specific_times is provided with window_size, this function will automatically
    load the required historical data (window_size timesteps before each target time)
    to enable prediction.
    
    Args:
        data_path: Path to NetCDF data file
        logger: Logger instance
        start_time: Optional start time for filtering (ISO format)
        end_time: Optional end time for filtering (ISO format)
        specific_times: Optional list of specific times to predict (ISO format)
        window_size: Number of historical timesteps needed for input (e.g., 6)
        target_offset: Number of timesteps ahead to predict (e.g., 1)
        
    Returns:
        Loaded xarray Dataset (potentially filtered by time)
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data file is invalid or time filtering fails
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Validate dates before loading data
    if start_time and not validate_date(start_time):
        raise ValueError(
            f"Invalid start_time: '{start_time}'. "
            f"Please use valid ISO format (e.g., 2020-06-30T00:00:00). "
            f"Note: June has only 30 days, not 31."
        )
    
    if end_time and not validate_date(end_time):
        raise ValueError(
            f"Invalid end_time: '{end_time}'. "
            f"Please use valid ISO format (e.g., 2020-06-30T23:00:00). "
            f"Note: June has only 30 days, not 31."
        )
    
    if specific_times:
        for time_str in specific_times:
            if not validate_date(time_str):
                raise ValueError(
                    f"Invalid time in specific_times: '{time_str}'. "
                    f"Please use valid ISO format (e.g., 2020-06-30T12:00:00). "
                    f"Note: June has only 30 days, not 31."
                )
    
    logger.info(f"Loading data from {data_path}")
    
    try:
        data = xr.open_dataset(data_path)
        logger.info(f"Data loaded successfully")
        logger.info(f"Data dimensions: {dict(data.dims)}")
        logger.info(f"Data variables: {list(data.data_vars)}")
        logger.info(f"Original time range: {data.time.values[0]} to {data.time.values[-1]}")
        logger.info(f"Total timesteps: {len(data.time)}")
        
        # Apply time filtering if specified
        if specific_times is not None:
            logger.info(f"Target prediction times: {specific_times}")
            
            # If window_size is provided, we need to load historical data
            if window_size is not None:
                logger.info(f"Auto-loading historical data (window_size={window_size})")
                
                # Convert specific_times to numpy datetime64 for easier manipulation
                target_times = []
                for time_str in specific_times:
                    try:
                        # Try exact match first
                        matched_time = data.sel(time=time_str, method='nearest').time.values
                        target_times.append(matched_time)
                        logger.info(f"  Target time: {time_str} -> Matched: {matched_time}")
                    except Exception as e:
                        raise ValueError(
                            f"Failed to find target time '{time_str}' in data. "
                            f"Available times (first 10): {[str(t) for t in data.time.values[:10]]}"
                        )
                
                # For each target time, find the required historical window
                all_required_times = set()
                
                for target_time in target_times:
                    # Find the index of target time
                    target_idx = np.where(data.time.values == target_time)[0]
                    
                    if len(target_idx) == 0:
                        raise ValueError(f"Target time {target_time} not found in data")
                    
                    target_idx = target_idx[0]
                    
                    # Calculate required historical window
                    # We need window_size timesteps for input + target_offset for prediction
                    # Total: window_size + target_offset timesteps
                    total_needed = window_size + (target_offset if target_offset else 1)
                    start_idx = target_idx - total_needed + 1
                    
                    if start_idx < 0:
                        raise ValueError(
                            f"Insufficient historical data for target time {target_time}. "
                            f"Need {total_needed} timesteps (window_size={window_size} + target_offset={target_offset if target_offset else 1}), "
                            f"but only {target_idx + 1} available. "
                            f"First available time: {data.time.values[0]}"
                        )
                    
                    # Add all required times (historical window + target)
                    for idx in range(start_idx, target_idx + 1):
                        all_required_times.add(data.time.values[idx])
                    
                    logger.info(
                        f"  For target {target_time}: "
                        f"loading {total_needed} timesteps from {data.time.values[start_idx]} "
                        f"to {data.time.values[target_idx]} "
                        f"(window_size={window_size} + target_offset={target_offset if target_offset else 1})"
                    )
                
                # Sort times and select from data
                required_times_sorted = sorted(list(all_required_times))
                logger.info(f"Total unique timesteps to load: {len(required_times_sorted)}")
                logger.info(f"Time range: {required_times_sorted[0]} to {required_times_sorted[-1]}")
                
                # Select the required times
                data = data.sel(time=required_times_sorted)
                logger.info(f"Loaded {len(data.time)} timesteps for prediction")
                
            else:
                # Original behavior: just select the specific times
                logger.info(f"Filtering to specific times: {specific_times}")
                try:
                    # Try exact match first
                    data = data.sel(time=specific_times)
                    logger.info(f"Selected {len(data.time)} specific timesteps")
                except KeyError as e:
                    # If exact match fails, try nearest neighbor matching
                    logger.warning(f"Exact time match failed, trying nearest neighbor matching...")
                    try:
                        data = data.sel(time=specific_times, method='nearest')
                        logger.info(f"Selected {len(data.time)} timesteps using nearest matching")
                        logger.info(f"Matched times: {[str(t) for t in data.time.values]}")
                    except Exception as e2:
                        available_times = [str(t) for t in data.time.values[:10]]
                        raise ValueError(
                            f"Failed to select specific times. Error: {e}\n"
                            f"Available times (first 10): {available_times}\n"
                            f"Requested times: {specific_times}\n"
                            f"Hint: Use format matching your data (e.g., 2019-05-08T02:00:00 or 2019-05-08T14:00:00)"
                        )
        elif start_time is not None or end_time is not None:
            logger.info(f"Filtering time range: {start_time} to {end_time}")
            try:
                data = data.sel(time=slice(start_time, end_time))
                logger.info(f"Filtered to {len(data.time)} timesteps")
            except Exception as e:
                available_times = [str(t) for t in data.time.values[:10]]
                raise ValueError(
                    f"Failed to filter time range: {e}\n"
                    f"Available times (first 10): {available_times}\n"
                    f"Hint: Check that your dates are valid and match the data format."
                )
        
        if len(data.time) == 0:
            raise ValueError("No data remaining after time filtering")
        
        logger.info(f"Final time range: {data.time.values[0]} to {data.time.values[-1]}")
        
        return data
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to load data file: {e}")


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run inference with trained ConvLSTM weather prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference - generate predictions
  python run_inference_convlstm.py \\
      --checkpoint checkpoints/baseline/best_model.pt \\
      --normalizer checkpoints/baseline/normalizer.pkl \\
      --data data/test_data.nc \\
      --output-dir outputs/baseline

  # Inference with visualizations
  python run_inference_convlstm.py \\
      --checkpoint checkpoints/baseline/best_model.pt \\
      --normalizer checkpoints/baseline/normalizer.pkl \\
      --data data/test_data.nc \\
      --output-dir outputs/baseline \\
      --visualize \\
      --viz-timesteps 0 5 10

  # Inference with upstream region
  python run_inference_convlstm.py \\
      --checkpoint checkpoints/with_upstream/best_model.pt \\
      --normalizer checkpoints/with_upstream/normalizer.pkl \\
      --data data/test_data.nc \\
      --output-dir outputs/with_upstream \\
      --include-upstream

  # Inference for specific time range
  python run_inference_convlstm.py \\
      --checkpoint checkpoints/baseline/best_model.pt \\
      --normalizer checkpoints/baseline/normalizer.pkl \\
      --data data/test_data.nc \\
      --output-dir outputs/baseline \\
      --start-time 2023-01-01T00:00:00 \\
      --end-time 2023-01-31T23:00:00

  # Inference for specific times
  python run_inference_convlstm.py \\
      --checkpoint checkpoints/baseline/best_model.pt \\
      --normalizer checkpoints/baseline/normalizer.pkl \\
      --data data/test_data.nc \\
      --output-dir outputs/baseline \\
      --specific-times 2023-01-15T12:00:00 2023-02-15T12:00:00

  # Inference with old checkpoint (manual model type specification)
  python run_inference_convlstm.py \\
      --checkpoint checkpoints/old_model.pt \\
      --normalizer checkpoints/normalizer.pkl \\
      --data data/test_data.nc \\
      --output-dir outputs/old_model \\
      --model-type dual_stream_deep

  # Compare two experiments
  python run_inference_convlstm.py \\
      --checkpoint checkpoints/baseline/best_model.pt \\
      --normalizer checkpoints/baseline/normalizer.pkl \\
      --data data/test_data.nc \\
      --output-dir outputs/comparison \\
      --compare-checkpoint checkpoints/with_upstream/best_model.pt \\
      --compare-normalizer checkpoints/with_upstream/normalizer.pkl \\
      --compare-upstream \\
      --visualize
        """
    )
    
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pt or .pth)'
    )
    required_group.add_argument(
        '--normalizer',
        type=str,
        required=True,
        help='Path to normalizer file (.pkl)'
    )
    required_group.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to input NetCDF data file'
    )
    required_group.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save predictions and visualizations'
    )
    
    # Region configuration
    region_group = parser.add_argument_group('Region Configuration')
    region_group.add_argument(
        '--include-upstream',
        action='store_true',
        help='Include upstream region in input (must match training configuration)'
    )
    region_group.add_argument(
        '--downstream-lat-min',
        type=float,
        default=25.0,
        help='Downstream region minimum latitude (default: 25.0)'
    )
    region_group.add_argument(
        '--downstream-lat-max',
        type=float,
        default=40.0,
        help='Downstream region maximum latitude (default: 40.0)'
    )
    region_group.add_argument(
        '--downstream-lon-min',
        type=float,
        default=110.0,
        help='Downstream region minimum longitude (default: 110.0)'
    )
    region_group.add_argument(
        '--downstream-lon-max',
        type=float,
        default=125.0,
        help='Downstream region maximum longitude (default: 125.0)'
    )
    region_group.add_argument(
        '--upstream-lat-min',
        type=float,
        default=25.0,
        help='Upstream region minimum latitude (default: 25.0)'
    )
    region_group.add_argument(
        '--upstream-lat-max',
        type=float,
        default=50.0,
        help='Upstream region maximum latitude (default: 50.0)'
    )
    region_group.add_argument(
        '--upstream-lon-min',
        type=float,
        default=70.0,
        help='Upstream region minimum longitude (default: 70.0)'
    )
    region_group.add_argument(
        '--upstream-lon-max',
        type=float,
        default=110.0,
        help='Upstream region maximum longitude (default: 110.0)'
    )
    
    # Inference configuration
    inference_group = parser.add_argument_group('Inference Configuration')
    inference_group.add_argument(
        '--model-type',
        type=str,
        default=None,
        choices=['shallow', 'deep', 'dual_stream', 'dual_stream_deep'],
        help='Model architecture type (optional, auto-detected from checkpoint if not specified). '
             'Use this to override auto-detection for old checkpoints without model_type metadata.'
    )
    inference_group.add_argument(
        '--window-size',
        type=int,
        default=6,
        help='Number of historical timesteps for input (default: 6)'
    )
    inference_group.add_argument(
        '--target-offset',
        type=int,
        default=1,
        help='Number of timesteps ahead to predict (default: 1)'
    )
    inference_group.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)'
    )
    
    # Time filtering
    time_group = parser.add_argument_group('Time Filtering')
    time_group.add_argument(
        '--start-time',
        type=str,
        default=None,
        help='Start time for predictions (ISO format: YYYY-MM-DDTHH:MM:SS)'
    )
    time_group.add_argument(
        '--end-time',
        type=str,
        default=None,
        help='End time for predictions (ISO format: YYYY-MM-DDTHH:MM:SS)'
    )
    time_group.add_argument(
        '--specific-times',
        type=str,
        nargs='+',
        default=None,
        help='Specific target times to predict (ISO format: YYYY-MM-DDTHH:MM:SS). '
             'Historical data (window_size timesteps) will be automatically loaded. '
             'Example: --specific-times 2023-06-15T14:00:00 will load data from '
             '2023-06-15T08:00:00 to 2023-06-15T14:00:00 (if window_size=6)'
    )
    
    # Comparison experiment
    comparison_group = parser.add_argument_group('Comparison Experiment')
    comparison_group.add_argument(
        '--compare-checkpoint',
        type=str,
        default=None,
        help='Path to second model checkpoint for comparison'
    )
    comparison_group.add_argument(
        '--compare-normalizer',
        type=str,
        default=None,
        help='Path to second normalizer for comparison'
    )
    comparison_group.add_argument(
        '--compare-upstream',
        action='store_true',
        help='Second model uses upstream region'
    )
    comparison_group.add_argument(
        '--compare-model-type',
        type=str,
        default=None,
        choices=['shallow', 'deep', 'dual_stream', 'dual_stream_deep'],
        help='Model type for comparison model (optional, auto-detected if not specified)'
    )
    comparison_group.add_argument(
        '--exp1-name',
        type=str,
        default='Experiment 1',
        help='Name for first experiment (default: Experiment 1)'
    )
    comparison_group.add_argument(
        '--exp2-name',
        type=str,
        default='Experiment 2',
        help='Name for second experiment (default: Experiment 2)'
    )
    
    # Visualization options
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    viz_group.add_argument(
        '--viz-timesteps',
        type=int,
        nargs='+',
        default=None,
        help='Specific timestep indices to visualize (default: first 5)'
    )
    viz_group.add_argument(
        '--viz-format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'jpg', 'svg'],
        help='Format for saved visualizations (default: png)'
    )
    viz_group.add_argument(
        '--viz-dpi',
        type=int,
        default=150,
        help='DPI for saved visualizations (default: 150)'
    )
    viz_group.add_argument(
        '--viz-vmin',
        type=float,
        default=0.0,
        help='Minimum value for precipitation color scale (default: 0.0 mm)'
    )
    viz_group.add_argument(
        '--viz-vmax',
        type=float,
        default=500.0,
        help='Maximum value for precipitation color scale (default: 500.0 mm)'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='Save predictions to NetCDF file (default: True)'
    )
    output_group.add_argument(
        '--no-save-predictions',
        action='store_false',
        dest='save_predictions',
        help='Do not save predictions to NetCDF file'
    )
    output_group.add_argument(
        '--predictions-filename',
        type=str,
        default='predictions.nc',
        help='Filename for saved predictions (default: predictions.nc)'
    )
    
    # Device and logging
    device_group = parser.add_argument_group('Device and Logging')
    device_group.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for inference (default: auto, mps for Apple Silicon Mac)'
    )
    device_group.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main inference workflow."""
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.log_level)
    
    logger.info("=" * 80)
    logger.info("ConvLSTM Weather Prediction Inference")
    logger.info("=" * 80)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Normalizer: {args.normalizer}")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Include upstream: {args.include_upstream}")
    logger.info(f"  Window size: {args.window_size}")
    logger.info(f"  Target offset: {args.target_offset}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    if args.start_time or args.end_time:
        logger.info(f"  Time range: {args.start_time} to {args.end_time}")
    if args.specific_times:
        logger.info(f"  Specific times: {args.specific_times}")
    
    if args.compare_checkpoint:
        logger.info(f"  Comparison mode enabled")
        logger.info(f"  Compare checkpoint: {args.compare_checkpoint}")
        logger.info(f"  Compare normalizer: {args.compare_normalizer}")
        logger.info(f"  Compare upstream: {args.compare_upstream}")
    
    # Set device
    if args.device == 'auto':
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"  Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        logger.info(f"  Apple Silicon GPU (MPS) detected")
        logger.info(f"  Note: MPS provides GPU acceleration on Mac")
    
    # Load data
    try:
        data = load_data(
            data_path=args.data,
            logger=logger,
            start_time=args.start_time,
            end_time=args.end_time,
            specific_times=args.specific_times,
            window_size=args.window_size,
            target_offset=args.target_offset
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Create region configuration
    region_config = RegionConfig(
        downstream_lat_min=args.downstream_lat_min,
        downstream_lat_max=args.downstream_lat_max,
        downstream_lon_min=args.downstream_lon_min,
        downstream_lon_max=args.downstream_lon_max,
        upstream_lat_min=args.upstream_lat_min,
        upstream_lat_max=args.upstream_lat_max,
        upstream_lon_min=args.upstream_lon_min,
        upstream_lon_max=args.upstream_lon_max
    )
    
    logger.info("Region configuration:")
    logger.info(f"  Downstream: lat=[{region_config.downstream_lat_min}, {region_config.downstream_lat_max}], "
                f"lon=[{region_config.downstream_lon_min}, {region_config.downstream_lon_max}]")
    if args.include_upstream:
        logger.info(f"  Upstream: lat=[{region_config.upstream_lat_min}, {region_config.upstream_lat_max}], "
                    f"lon=[{region_config.upstream_lon_min}, {region_config.upstream_lon_max}]")
    
    # Load model and normalizer
    logger.info("Loading model and normalizer...")
    if args.model_type:
        logger.info(f"Using manually specified model type: {args.model_type}")
    try:
        model, checkpoint_data, model_type = load_trained_model_auto(
            args.checkpoint, 
            device=device,
            model_type_override=args.model_type
        )
        normalizer = load_normalizer(args.normalizer)
        
        logger.info(f"Model type: {model_type}")
        logger.info(f"Model architecture: {checkpoint_data['model_architecture']}")
        logger.info(f"Model loaded from epoch {checkpoint_data['epoch']}")
        logger.info(f"Best validation loss: {checkpoint_data['best_val_loss']:.4f}")
        
        # Auto-detect if upstream region should be used
        is_dual_stream_model = model_type in ['dual_stream', 'dual_stream_deep']
        
        if is_dual_stream_model and not args.include_upstream:
            logger.warning("=" * 80)
            logger.warning("IMPORTANT: Dual-stream model detected!")
            logger.warning(f"  Model type: {model_type}")
            logger.warning("  This model was trained with upstream region data.")
            logger.warning("  Automatically enabling --include-upstream for inference.")
            logger.warning("=" * 80)
            args.include_upstream = True
        elif not is_dual_stream_model and args.include_upstream:
            logger.warning("=" * 80)
            logger.warning("WARNING: Single-stream model detected!")
            logger.warning(f"  Model type: {model_type}")
            logger.warning("  This model was trained WITHOUT upstream region data.")
            logger.warning("  The --include-upstream flag will be ignored.")
            logger.warning("  Only downstream region will be used for prediction.")
            logger.warning("=" * 80)
            args.include_upstream = False
        
        logger.info(f"Using upstream region: {args.include_upstream}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
    except Exception as e:
        logger.error(f"Failed to load model or normalizer: {e}")
        sys.exit(1)
    
    # Load comparison model if specified
    model2 = None
    normalizer2 = None
    if args.compare_checkpoint:
        logger.info("Loading comparison model and normalizer...")
        if args.compare_model_type:
            logger.info(f"Using manually specified comparison model type: {args.compare_model_type}")
        try:
            model2, checkpoint_data2, model_type2 = load_trained_model_auto(
                args.compare_checkpoint, 
                device=device,
                model_type_override=args.compare_model_type
            )
            normalizer2 = load_normalizer(args.compare_normalizer)
            
            logger.info(f"Comparison model type: {model_type2}")
            logger.info(f"Comparison model loaded from epoch {checkpoint_data2['epoch']}")
            logger.info(f"Best validation loss: {checkpoint_data2['best_val_loss']:.4f}")
        except Exception as e:
            logger.error(f"Failed to load comparison model or normalizer: {e}")
            sys.exit(1)
    
    # Generate predictions
    logger.info("Generating predictions...")
    logger.info("=" * 80)
    
    try:
        predictions = predict_batch(
            model=model,
            input_data=data,
            normalizer=normalizer,
            region_config=region_config,
            window_size=args.window_size,
            target_offset=args.target_offset,
            include_upstream=args.include_upstream,
            batch_size=args.batch_size,
            device=device
        )
        
        logger.info(f"Generated {len(predictions.time)} predictions")
        logger.info(f"Prediction shape: {predictions.precipitation.shape}")
        
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
    
    # Generate comparison predictions if specified
    predictions2 = None
    if model2 is not None:
        logger.info("Generating comparison predictions...")
        try:
            predictions2 = predict_batch(
                model=model2,
                input_data=data,
                normalizer=normalizer2,
                region_config=region_config,
                window_size=args.window_size,
                target_offset=args.target_offset,
                include_upstream=args.compare_upstream,
                batch_size=args.batch_size,
                device=device
            )
            
            logger.info(f"Generated {len(predictions2.time)} comparison predictions")
            
        except Exception as e:
            logger.error(f"Failed to generate comparison predictions: {e}")
            logger.exception("Full traceback:")
            sys.exit(1)
    
    # Save predictions to NetCDF
    if args.save_predictions:
        logger.info("Saving predictions to NetCDF...")
        try:
            predictions_path = output_dir / args.predictions_filename
            predictions.to_netcdf(predictions_path)
            logger.info(f"Predictions saved to {predictions_path}")
            
            if predictions2 is not None:
                predictions2_path = output_dir / f"comparison_{args.predictions_filename}"
                predictions2.to_netcdf(predictions2_path)
                logger.info(f"Comparison predictions saved to {predictions2_path}")
                
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            logger.exception("Full traceback:")
    
    # Generate visualizations
    if args.visualize:
        logger.info("=" * 80)
        logger.info("Generating visualizations...")
        
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which timesteps to visualize
        if args.viz_timesteps is not None:
            viz_timesteps = args.viz_timesteps
        else:
            # Default: first 5 timesteps
            viz_timesteps = list(range(min(5, len(predictions.time))))
        
        logger.info(f"Visualizing timesteps: {viz_timesteps}")
        
        # Extract ground truth for comparison
        # Ground truth is at target_offset ahead of the last input timestep
        downstream_data = data.sel(
            lat=slice(region_config.downstream_lat_min, region_config.downstream_lat_max),
            lon=slice(region_config.downstream_lon_min, region_config.downstream_lon_max)
        )
        
        for idx in viz_timesteps:
            if idx >= len(predictions.time):
                logger.warning(f"Timestep {idx} out of range, skipping")
                continue
            
            try:
                # Get prediction for this timestep
                pred = predictions.isel(time=idx)
                pred_time = predictions.time.values[idx]
                
                logger.info(f"Visualizing timestep {idx} ({pred_time})")
                
                # Find corresponding ground truth
                # The prediction time should match a time in the original data
                try:
                    truth = downstream_data.sel(time=pred_time)
                except KeyError:
                    logger.warning(f"No ground truth found for time {pred_time}, skipping")
                    continue
                
                # Create side-by-side comparison: Ground Truth vs Prediction
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Get actual max values for display
                truth_max = np.nanmax(truth.precipitation.values)
                pred_max = np.nanmax(pred.precipitation.values)
                
                # Use unified color scale from command line arguments
                vmin = args.viz_vmin
                vmax = args.viz_vmax
                
                # Plot ground truth with unified color scale
                plot_precipitation_map(
                    precipitation=truth.precipitation,
                    title=f"Ground Truth - {pred_time}\n(Max: {truth_max:.2f} mm, Scale: {vmin}-{vmax} mm)",
                    ax=axes[0],
                    vmin=vmin,
                    vmax=vmax,
                    show_colorbar=True
                )
                
                # Plot prediction with unified color scale
                plot_precipitation_map(
                    precipitation=pred.precipitation,
                    title=f"Prediction - {pred_time}\n(Max: {pred_max:.2f} mm, Scale: {vmin}-{vmax} mm)",
                    ax=axes[1],
                    vmin=vmin,
                    vmax=vmax,
                    show_colorbar=True
                )
                
                # Add overall title
                fig.suptitle(f"Precipitation Comparison - {pred_time}", 
                           fontsize=16, fontweight='bold', y=1.00)
                
                plt.tight_layout()
                
                save_plot(
                    fig=fig,
                    output_dir=viz_dir,
                    experiment_name=args.exp1_name.replace(' ', '_'),
                    timestep=idx,
                    plot_type='truth_vs_pred',
                    dpi=args.viz_dpi,
                    format=args.viz_format
                )
                plt.close(fig)
                
                # Create error map
                fig, ax = plt.subplots(figsize=(10, 8))
                create_error_map(
                    prediction=pred.precipitation,
                    ground_truth=truth.precipitation,
                    title=f"Prediction Error - {pred_time}",
                    ax=ax
                )
                save_plot(
                    fig=fig,
                    output_dir=viz_dir,
                    experiment_name=args.exp1_name.replace(' ', '_'),
                    timestep=idx,
                    plot_type='error',
                    dpi=args.viz_dpi,
                    format=args.viz_format
                )
                plt.close(fig)
                
                # Create comparison plot if second model exists
                if predictions2 is not None:
                    pred2 = predictions2.isel(time=idx)
                    
                    fig = create_comparison_plot(
                        ground_truth=truth.precipitation,
                        experiment1=pred.precipitation,
                        experiment2=pred2.precipitation,
                        exp1_name=args.exp1_name,
                        exp2_name=args.exp2_name,
                        title=f"Precipitation Comparison - {pred_time}"
                    )
                    save_plot(
                        fig=fig,
                        output_dir=viz_dir,
                        experiment_name='comparison',
                        timestep=idx,
                        plot_type='comparison',
                        dpi=args.viz_dpi,
                        format=args.viz_format
                    )
                    plt.close(fig)
                    
                    # Create multi-error comparison
                    fig = create_multi_error_comparison(
                        predictions_list=[pred.precipitation, pred2.precipitation],
                        ground_truth=truth.precipitation,
                        experiment_names=[args.exp1_name, args.exp2_name],
                        title=f"Error Comparison - {pred_time}"
                    )
                    save_plot(
                        fig=fig,
                        output_dir=viz_dir,
                        experiment_name='error_comparison',
                        timestep=idx,
                        plot_type='multi_error',
                        dpi=args.viz_dpi,
                        format=args.viz_format
                    )
                    plt.close(fig)
                
                logger.info(f"Visualizations saved for timestep {idx}")
                
            except Exception as e:
                logger.error(f"Failed to create visualizations for timestep {idx}: {e}")
                logger.exception("Full traceback:")
                continue
        
        logger.info(f"All visualizations saved to {viz_dir}")
    
    logger.info("=" * 80)
    logger.info("Inference completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    if args.save_predictions:
        logger.info(f"  Predictions: {output_dir / args.predictions_filename}")
    if args.visualize:
        logger.info(f"  Visualizations: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
