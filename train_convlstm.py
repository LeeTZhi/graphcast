#!/usr/bin/env python3
"""Training script for ConvLSTM weather prediction model.

This script provides a comprehensive CLI for training ConvLSTM models with
support for:
- Configurable model architecture and training hyperparameters
- Upstream region inclusion for comparative experiments
- Memory optimization (mixed precision, gradient accumulation)
- Checkpoint saving and resumption
- Comprehensive logging

Example usage:
    # Train baseline model (downstream only)
    python train_convlstm.py --data data/regional_weather.nc \\
        --output-dir checkpoints/baseline

    # Train with upstream region
    python train_convlstm.py --data data/regional_weather.nc \\
        --output-dir checkpoints/with_upstream \\
        --include-upstream

    # Resume training from checkpoint
    python train_convlstm.py --data data/regional_weather.nc \\
        --output-dir checkpoints/baseline \\
        --resume checkpoints/baseline/checkpoint_epoch_10.pt

    # Train with custom hyperparameters
    python train_convlstm.py --data data/regional_weather.nc \\
        --output-dir checkpoints/custom \\
        --hidden-channels 64 128 \\
        --batch-size 8 \\
        --learning-rate 5e-4 \\
        --num-epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray as xr

from convlstm.model import ConvLSTMUNet, WeightedPrecipitationLoss
from convlstm.model_deep import DeepConvLSTMUNet
from convlstm.model_dual_stream import DualStreamConvLSTMUNet
from convlstm.model_dual_stream_deep import DeepDualStreamConvLSTMUNet
from convlstm.config import ConvLSTMConfig
from convlstm.data import (
    ConvLSTMDataset,
    ConvLSTMNormalizer,
    RegionConfig,
    create_train_val_test_split
)
from convlstm.trainer import ConvLSTMTrainer


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
    log_file = output_dir / "training.log"
    
    # Create logger
    logger = logging.getLogger("train_convlstm")
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


def load_data(data_path: str, logger: logging.Logger) -> xr.Dataset:
    """Load xarray dataset from file.
    
    Args:
        data_path: Path to NetCDF data file
        logger: Logger instance
        
    Returns:
        Loaded xarray Dataset
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data file is invalid
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    
    try:
        data = xr.open_dataset(data_path)
        logger.info(f"Data loaded successfully")
        logger.info(f"Data dimensions: {dict(data.sizes)}")
        logger.info(f"Data variables: {list(data.data_vars)}")
        logger.info(f"Time range: {data.time.values[0]} to {data.time.values[-1]}")
        
        return data
    except Exception as e:
        raise ValueError(f"Failed to load data file: {e}")


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train ConvLSTM weather prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline model (downstream only)
  python train_convlstm.py --data data/regional_weather.nc \\
      --output-dir checkpoints/baseline

  # Train with upstream region
  python train_convlstm.py --data data/regional_weather.nc \\
      --output-dir checkpoints/with_upstream \\
      --include-upstream

  # Resume training from checkpoint
  python train_convlstm.py --data data/regional_weather.nc \\
      --output-dir checkpoints/baseline \\
      --resume checkpoints/baseline/checkpoint_epoch_10.pt

  # Train with custom hyperparameters
  python train_convlstm.py --data data/regional_weather.nc \\
      --output-dir checkpoints/custom \\
      --hidden-channels 64 128 \\
      --batch-size 8 \\
      --learning-rate 5e-4 \\
      --num-epochs 50

  # Memory-optimized training for 12GB GPU
  python train_convlstm.py --data data/regional_weather.nc \\
      --output-dir checkpoints/memory_opt \\
      --batch-size 4 \\
      --gradient-accumulation-steps 2 \\
      --use-amp

  # Train with specific time cutoff for train/val data
  python train_convlstm.py --data data/regional_weather.nc \\
      --output-dir checkpoints/time_split \\
      --trainval-end-date 2020-01-01 \\
      --train-ratio 0.85 \\
      --val-ratio 0.15
        """
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to NetCDF data file'
    )
    data_group.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save checkpoints and logs'
    )
    
    # Region configuration
    region_group = parser.add_argument_group('Region Configuration')
    region_group.add_argument(
        '--include-upstream',
        action='store_true',
        help='Include upstream region in input (for comparative experiments)'
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
    
    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        '--model-type',
        type=str,
        default='dual_stream',
        choices=['shallow', 'deep', 'dual_stream', 'dual_stream_deep'],
        help='Model architecture type: shallow (2 layers), deep (4 layers), dual_stream (2 layers with dual-stream), or dual_stream_deep (4 layers with dual-stream) (default: dual_stream)'
    )
    model_group.add_argument(
        '--hidden-channels',
        type=int,
        nargs='+',
        default=None,
        help='Hidden channel dimensions. If not specified, uses default for model type. '
             'Shallow: [32, 64], Deep: [64, 128, 256, 512]'
    )
    model_group.add_argument(
        '--kernel-size',
        type=int,
        default=3,
        help='Convolutional kernel size (default: 3)'
    )
    
    # Regularization
    regularization_group = parser.add_argument_group('Regularization (for Deep model)')
    regularization_group.add_argument(
        '--dropout-rate',
        type=float,
        default=0.2,
        help='Dropout rate for regularization (default: 0.2, set to 0 to disable)'
    )
    regularization_group.add_argument(
        '--use-batch-norm',
        action='store_true',
        default=False,
        help='Use batch normalization (default: False, not recommended for small batches)'
    )
    regularization_group.add_argument(
        '--no-batch-norm',
        action='store_false',
        dest='use_batch_norm',
        help='Disable batch normalization'
    )
    regularization_group.add_argument(
        '--use-group-norm',
        action='store_true',
        default=True,
        help='Use group normalization in ConvLSTM cells (default: True, recommended)'
    )
    regularization_group.add_argument(
        '--no-group-norm',
        action='store_false',
        dest='use_group_norm',
        help='Disable group normalization'
    )
    regularization_group.add_argument(
        '--use-spatial-dropout',
        action='store_true',
        default=True,
        help='Use spatial dropout (default: True)'
    )
    regularization_group.add_argument(
        '--no-spatial-dropout',
        action='store_false',
        dest='use_spatial_dropout',
        help='Use regular dropout instead of spatial dropout'
    )
    regularization_group.add_argument(
        '--use-attention',
        action='store_true',
        default=True,
        help='Use self-attention at bottleneck (default: True, recommended)'
    )
    regularization_group.add_argument(
        '--no-attention',
        action='store_false',
        dest='use_attention',
        help='Disable self-attention mechanism'
    )
    
    # Training configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Initial learning rate (default: 1e-4)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    train_group.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    train_group.add_argument(
        '--gradient-clip-norm',
        type=float,
        default=1.0,
        help='Gradient clipping norm (default: 1.0)'
    )
    train_group.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay for AdamW optimizer (default: 1e-5)'
    )
    
    # Data configuration
    data_config_group = parser.add_argument_group('Data Processing')
    data_config_group.add_argument(
        '--window-size',
        type=int,
        default=6,
        help='Number of historical timesteps for input (default: 6)'
    )
    data_config_group.add_argument(
        '--target-offset',
        type=int,
        default=1,
        help='Number of timesteps ahead to predict (default: 1)'
    )
    data_config_group.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of data for training (default: 0.7)'
    )
    data_config_group.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Fraction of data for validation (default: 0.15)'
    )
    data_config_group.add_argument(
        '--trainval-end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD) for train/val data. Data before this date is used for '
             'training and validation, data from this date onwards is used for testing. '
             'If not specified, uses ratio-based splitting for all data.'
    )
    data_config_group.add_argument(
        '--test-start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD) for test data. Alternative to trainval-end-date. '
             'Data before this date is split into train/val using ratios.'
    )
    
    # Loss function
    loss_group = parser.add_argument_group('Loss Function')
    loss_group.add_argument(
        '--high-precip-threshold',
        type=float,
        default=10.0,
        help='Precipitation threshold (mm) for high-weight events (default: 10.0)'
    )
    loss_group.add_argument(
        '--high-precip-weight',
        type=float,
        default=3.0,
        help='Weight multiplier for high precipitation events (default: 3.0)'
    )
    
    # Memory optimization
    memory_group = parser.add_argument_group('Memory Optimization')
    memory_group.add_argument(
        '--use-amp',
        action='store_true',
        default=True,
        help='Use automatic mixed precision training (default: True)'
    )
    memory_group.add_argument(
        '--no-amp',
        action='store_false',
        dest='use_amp',
        help='Disable automatic mixed precision training'
    )
    memory_group.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps (default: 1)'
    )
    memory_group.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of DataLoader worker processes (default: 2)'
    )
    
    # Checkpointing
    checkpoint_group = parser.add_argument_group('Checkpointing')
    checkpoint_group.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=1000,
        help='Save checkpoint every N training steps (default: 1000)'
    )
    checkpoint_group.add_argument(
        '--validation-frequency',
        type=int,
        default=500,
        help='Run validation every N training steps (default: 500)'
    )
    checkpoint_group.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Stop training after N epochs without improvement (default: 10)'
    )
    checkpoint_group.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    checkpoint_group.add_argument(
        '--reset-lr-on-resume',
        action='store_true',
        help='Reset learning rate to command line value when resuming (default: use checkpoint LR)'
    )
    
    # Logging
    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    # Device
    device_group = parser.add_argument_group('Device')
    device_group.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for training (default: auto, mps for Apple Silicon Mac)'
    )
    
    return parser.parse_args()


def main():
    """Main training workflow."""
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.log_level)
    
    logger.info("=" * 80)
    logger.info("ConvLSTM Weather Prediction Training")
    logger.info("=" * 80)
    
    # Set default hidden channels based on model type if not specified
    if args.hidden_channels is None:
        if args.model_type == 'shallow':
            args.hidden_channels = [32, 64]
        elif args.model_type == 'dual_stream':
            args.hidden_channels = [64, 128]
        else:  # deep or dual_stream_deep
            args.hidden_channels = [64, 128, 256, 512]
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Include upstream: {args.include_upstream}")
    logger.info(f"  Hidden channels: {args.hidden_channels}")
    logger.info(f"  Self-attention: {args.use_attention}")
    logger.info(f"  Group normalization: {args.use_group_norm}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Number of epochs: {args.num_epochs}")
    logger.info(f"  Mixed precision: {args.use_amp}")
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
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
        logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == 'mps':
        logger.info(f"  Apple Silicon GPU (MPS) detected")
        logger.info(f"  Note: MPS provides GPU acceleration on Mac")
    
    # Load data
    try:
        data = load_data(args.data, logger)
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
    
    # Create and fit normalizer (with caching)
    # Note: We need to determine train/test split first to fit normalizer only on training data
    normalizer_path = output_dir / "normalizer.pkl"
    normalizer = ConvLSTMNormalizer()
    
    # Determine test cutoff for normalizer fitting
    logger.info("Determining train/test split for normalization...")
    times = data.time.values
    
    if args.trainval_end_date is not None:
        test_cutoff = np.datetime64(args.trainval_end_date)
        logger.info(f"Using trainval_end_date: {args.trainval_end_date}")
    elif args.test_start_date is not None:
        test_cutoff = np.datetime64(args.test_start_date)
        logger.info(f"Using test_start_date: {args.test_start_date}")
    else:
        # Use ratio-based splitting
        n_total = len(times)
        n_trainval = int(n_total * (args.train_ratio + args.val_ratio))
        test_cutoff = times[n_trainval]
        logger.info(f"Using ratio-based split, test cutoff: {test_cutoff}")
    
    # Get trainval data for normalizer fitting
    trainval_mask = times < test_cutoff
    trainval_data_for_norm = data.isel(time=trainval_mask)
    
    # Check if normalizer already exists (from previous run or resume)
    if normalizer_path.exists():
        logger.info(f"Found existing normalizer at {normalizer_path}")
        logger.info("Loading cached normalization statistics...")
        try:
            normalizer.load(str(normalizer_path))
            logger.info("Normalizer loaded successfully (skipped recomputation)")
            if args.resume is not None:
                logger.info("Using normalizer from checkpoint directory for resumed training")
        except Exception as e:
            logger.warning(f"Failed to load cached normalizer: {e}")
            logger.info("Computing normalization statistics from scratch...")
            try:
                normalizer.fit(trainval_data_for_norm)
                normalizer.save(str(normalizer_path))
                logger.info(f"Normalizer saved to {normalizer_path}")
            except Exception as e:
                logger.error(f"Failed to fit normalizer: {e}")
                sys.exit(1)
    else:
        logger.info("Computing normalization statistics from train+val data...")
        try:
            normalizer.fit(trainval_data_for_norm)
            normalizer.save(str(normalizer_path))
            logger.info(f"Normalizer saved to {normalizer_path}")
        except Exception as e:
            logger.error(f"Failed to fit normalizer: {e}")
            sys.exit(1)
    
    # Normalize entire dataset BEFORE splitting
    logger.info("Normalizing entire dataset (before splitting)...")
    try:
        data_normalized = normalizer.normalize(data)
        logger.info("Data normalized successfully")
    except Exception as e:
        logger.error(f"Failed to normalize data: {e}")
        sys.exit(1)
    
    # Split normalized data into train/val/test
    logger.info("Splitting normalized data into train/val/test sets...")
    try:
        train_data_norm, val_data_norm, test_data_norm = create_train_val_test_split(
            data_normalized,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            trainval_end_date=args.trainval_end_date,
            test_start_date=args.test_start_date,
            random_trainval_split=False  # 使用时序划分，避免数据泄露
        )
        logger.info(f"Train: {len(train_data_norm.time)} timesteps")
        logger.info(f"Val: {len(val_data_norm.time)} timesteps")
        logger.info(f"Test: {len(test_data_norm.time)} timesteps")
        
        # Log time ranges for each split
        if len(train_data_norm.time) > 0:
            logger.info(f"Train time range: {train_data_norm.time.values[0]} to {train_data_norm.time.values[-1]}")
        if len(val_data_norm.time) > 0:
            logger.info(f"Val time range: {val_data_norm.time.values[0]} to {val_data_norm.time.values[-1]}")
        if len(test_data_norm.time) > 0:
            logger.info(f"Test time range: {test_data_norm.time.values[0]} to {test_data_norm.time.values[-1]}")
            
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        sys.exit(1)
    
    # Create datasets
    logger.info("Creating datasets...")
    try:
        train_dataset = ConvLSTMDataset(
            data=train_data_norm,
            window_size=args.window_size,
            region_config=region_config,
            target_offset=args.target_offset,
            include_upstream=args.include_upstream
        )
        
        val_dataset = ConvLSTMDataset(
            data=val_data_norm,
            window_size=args.window_size,
            region_config=region_config,
            target_offset=args.target_offset,
            include_upstream=args.include_upstream
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        sys.exit(1)
    
    # Create model configuration
    config = ConvLSTMConfig(
        input_channels=56,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        output_channels=1,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_clip_norm=args.gradient_clip_norm,
        weight_decay=args.weight_decay,
        window_size=args.window_size,
        target_offset=args.target_offset,
        high_precip_threshold=args.high_precip_threshold,
        high_precip_weight=args.high_precip_weight,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        checkpoint_frequency=args.checkpoint_frequency,
        validation_frequency=args.validation_frequency,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Validate configuration
    try:
        config.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)
    
    # Create model based on model type
    logger.info("Creating model...")
    try:
        if args.model_type == 'shallow':
            model = ConvLSTMUNet(
                input_channels=config.input_channels,
                hidden_channels=config.hidden_channels,
                output_channels=config.output_channels,
                kernel_size=config.kernel_size,
                use_attention=args.use_attention,
                use_group_norm=args.use_group_norm
            )
            logger.info(f"Shallow model features: attention={args.use_attention}, "
                       f"group_norm={args.use_group_norm}")
        elif args.model_type == 'deep':
            model = DeepConvLSTMUNet(
                input_channels=config.input_channels,
                hidden_channels=config.hidden_channels,
                output_channels=config.output_channels,
                kernel_size=config.kernel_size,
                dropout_rate=args.dropout_rate,
                use_batch_norm=args.use_batch_norm,
                use_group_norm=args.use_group_norm,
                use_spatial_dropout=args.use_spatial_dropout,
                use_attention=args.use_attention
            )
            logger.info(f"Deep model regularization: dropout={args.dropout_rate}, "
                       f"batch_norm={args.use_batch_norm}, "
                       f"group_norm={args.use_group_norm}, "
                       f"spatial_dropout={args.use_spatial_dropout}, "
                       f"attention={args.use_attention}")
        elif args.model_type == 'dual_stream':
            model = DualStreamConvLSTMUNet(
                input_channels=config.input_channels,
                hidden_channels=config.hidden_channels,
                output_channels=config.output_channels,
                kernel_size=config.kernel_size,
                use_attention=args.use_attention,
                use_group_norm=args.use_group_norm,
                dropout_rate=args.dropout_rate
            )
            logger.info(f"Dual-stream model features: attention={args.use_attention}, "
                       f"group_norm={args.use_group_norm}, dropout={args.dropout_rate}")
        elif args.model_type == 'dual_stream_deep':
            model = DeepDualStreamConvLSTMUNet(
                input_channels=config.input_channels,
                hidden_channels=config.hidden_channels,
                output_channels=config.output_channels,
                kernel_size=config.kernel_size,
                use_attention=args.use_attention,
                use_group_norm=args.use_group_norm,
                dropout_rate=args.dropout_rate
            )
            logger.info(f"Deep dual-stream model features: attention={args.use_attention}, "
                       f"group_norm={args.use_group_norm}, dropout={args.dropout_rate}")
        else:
            raise ValueError(f"Unknown model_type: {args.model_type}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created ({args.model_type}): {num_params:,} parameters ({num_trainable:,} trainable)")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)
    
    # Create trainer
    logger.info("Creating trainer...")
    try:
        trainer = ConvLSTMTrainer(
            model=model,
            config=config,
            region_config=region_config,
            normalizer=normalizer,
            device=device,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        sys.exit(1)
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            trainer.load_checkpoint(Path(args.resume))
            logger.info("Checkpoint loaded successfully")
            
            # Reset learning rate if requested or if non-default LR specified
            current_lr = trainer.optimizer.param_groups[0]['lr']
            should_reset = args.reset_lr_on_resume or (args.learning_rate != 1e-4)
            
            if should_reset:
                logger.info(f"Overriding checkpoint learning rate ({current_lr:.6f}) "
                           f"with command line value ({args.learning_rate:.6f})")
                trainer.reset_learning_rate(args.learning_rate)
            else:
                logger.info(f"Using checkpoint learning rate: {current_lr:.6f}")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)
    
    # Train model
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    try:
        results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_dir=output_dir
        )
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"Total training time: {results['total_time']:.2f}s")
        logger.info(f"Checkpoints saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(output_dir / "interrupted_checkpoint.pt")
        logger.info("Checkpoint saved")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
