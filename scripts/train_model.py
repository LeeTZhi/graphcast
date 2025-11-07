#!/usr/bin/env python3
# Copyright 2024 Regional Weather Prediction Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training script for Regional Weather Prediction System.

This script trains the Regional GNN model using preprocessed NetCDF data.

Example usage:
    python scripts/train_model.py \\
        --data data/processed/regional_weather.nc \\
        --output-dir checkpoints/experiment_1 \\
        --latent-size 256 \\
        --num-gnn-layers 12 \\
        --learning-rate 1e-4 \\
        --batch-size 4 \\
        --num-epochs 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import xarray as xr

from graphcast_regional.config import ModelConfig, RegionConfig, TrainingConfig
from graphcast_regional.graph_builder import RegionalGraphBuilder
from graphcast_regional.training import TrainingPipeline
from graphcast_regional import types


def setup_logging(output_dir: str, verbose: bool = False):
    """Configure logging for the script.
    
    Args:
        output_dir: Directory to save log file.
        verbose: If True, set logging level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging to both file and console
    log_file = Path(output_dir) / "training.log"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Regional Weather Prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to preprocessed NetCDF dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints',
        help='Directory for saving checkpoints and logs'
    )
    
    # Region configuration
    parser.add_argument(
        '--downstream-lat-min',
        type=float,
        default=25.0,
        help='Minimum latitude for downstream region'
    )
    parser.add_argument(
        '--downstream-lat-max',
        type=float,
        default=40.0,
        help='Maximum latitude for downstream region'
    )
    parser.add_argument(
        '--downstream-lon-min',
        type=float,
        default=110.0,
        help='Minimum longitude for downstream region'
    )
    parser.add_argument(
        '--downstream-lon-max',
        type=float,
        default=125.0,
        help='Maximum longitude for downstream region'
    )
    parser.add_argument(
        '--upstream-lat-min',
        type=float,
        default=25.0,
        help='Minimum latitude for upstream region'
    )
    parser.add_argument(
        '--upstream-lat-max',
        type=float,
        default=50.0,
        help='Maximum latitude for upstream region'
    )
    parser.add_argument(
        '--upstream-lon-min',
        type=float,
        default=70.0,
        help='Minimum longitude for upstream region'
    )
    parser.add_argument(
        '--upstream-lon-max',
        type=float,
        default=110.0,
        help='Maximum longitude for upstream region'
    )
    
    # Model architecture
    parser.add_argument(
        '--latent-size',
        type=int,
        default=256,
        help='Dimension of latent node representations'
    )
    parser.add_argument(
        '--num-gnn-layers',
        type=int,
        default=12,
        help='Number of message passing layers'
    )
    parser.add_argument(
        '--mlp-hidden-size',
        type=int,
        default=256,
        help='Hidden layer size for MLPs'
    )
    parser.add_argument(
        '--mlp-num-hidden-layers',
        type=int,
        default=2,
        help='Number of hidden layers in MLPs'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--gradient-clip-norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay coefficient'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=1000,
        help='Number of warmup steps for learning rate'
    )
    
    # Loss function
    parser.add_argument(
        '--high-precip-threshold',
        type=float,
        default=10.0,
        help='Threshold (mm) for high precipitation weighting'
    )
    parser.add_argument(
        '--high-precip-weight',
        type=float,
        default=3.0,
        help='Weight multiplier for high precipitation'
    )
    
    # Training loop
    parser.add_argument(
        '--validation-frequency',
        type=int,
        default=500,
        help='Validate every N training steps'
    )
    parser.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=1000,
        help='Save checkpoint every N training steps'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Stop if no improvement for N validations'
    )
    
    # Data splitting
    parser.add_argument(
        '--train-end-year',
        type=int,
        default=None,
        help='Last year (inclusive) for training data (optional, uses ratios if not set)'
    )
    parser.add_argument(
        '--val-end-year',
        type=int,
        default=None,
        help='Last year (inclusive) for validation data (optional, uses ratios if not set)'
    )
    parser.add_argument(
        '--test-start-date',
        type=str,
        default='2020-06-01',
        help='Date (YYYY-MM-DD) when test set starts. Data before this is split into train/val.'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of train+val data for training (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.3,
        help='Fraction of train+val data for validation (default: 0.3)'
    )
    
    # GPU optimization options
    parser.add_argument(
        '--use-prefetch',
        action='store_true',
        default=True,
        help='Enable data prefetching for GPU (default: True)'
    )
    parser.add_argument(
        '--no-prefetch',
        action='store_false',
        dest='use_prefetch',
        help='Disable data prefetching'
    )
    parser.add_argument(
        '--prefetch-buffer-size',
        type=int,
        default=4,
        help='Number of samples to prefetch (default: 4)'
    )
    parser.add_argument(
        '--jax-platform',
        type=str,
        default=None,
        choices=['cpu', 'gpu', 'tpu'],
        help='JAX platform to use (default: auto-detect)'
    )
    parser.add_argument(
        '--xla-flags',
        type=str,
        default=None,
        help='Additional XLA compiler flags (e.g., "--xla_gpu_cuda_data_dir=/usr/local/cuda")'
    )
    
    # Resume training
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    
    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    return parser.parse_args()


def main():
    """Main training workflow."""
    args = parse_args()
    
    # Configure JAX platform if specified
    if args.jax_platform:
        import os
        os.environ['JAX_PLATFORM_NAME'] = args.jax_platform
        logger.info(f"JAX platform set to: {args.jax_platform}")
    
    # Set XLA flags if specified
    if args.xla_flags:
        import os
        os.environ['XLA_FLAGS'] = args.xla_flags
        logger.info(f"XLA flags set to: {args.xla_flags}")
    
    # Import JAX after environment variables are set
    import jax
    
    setup_logging(args.output_dir, args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Regional Weather Prediction Model Training")
    logger.info("=" * 80)
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"JAX devices: {jax.devices()}")
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Model: latent_size={args.latent_size}, num_gnn_layers={args.num_gnn_layers}")
    logger.info(f"  Training: lr={args.learning_rate}, batch_size={args.batch_size}, "
               f"epochs={args.num_epochs}")
    logger.info(f"  Data split: train<={args.train_end_year}, val<={args.val_end_year}")
    logger.info("")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Create configurations
        logger.info("Creating configurations...")
        
        region_config = RegionConfig(
            downstream_lat_min=args.downstream_lat_min,
            downstream_lat_max=args.downstream_lat_max,
            downstream_lon_min=args.downstream_lon_min,
            downstream_lon_max=args.downstream_lon_max,
            upstream_lat_min=args.upstream_lat_min,
            upstream_lat_max=args.upstream_lat_max,
            upstream_lon_min=args.upstream_lon_min,
            upstream_lon_max=args.upstream_lon_max,
        )
        
        model_config = ModelConfig(
            latent_size=args.latent_size,
            num_gnn_layers=args.num_gnn_layers,
            mlp_hidden_size=args.mlp_hidden_size,
            mlp_num_hidden_layers=args.mlp_num_hidden_layers,
        )
        
        training_config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            gradient_clip_norm=args.gradient_clip_norm,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            high_precip_threshold=args.high_precip_threshold,
            high_precip_weight=args.high_precip_weight,
            validation_frequency=args.validation_frequency,
            checkpoint_frequency=args.checkpoint_frequency,
            early_stopping_patience=args.early_stopping_patience,
        )
        
        logger.info("Configurations created successfully")
        logger.info("")
        
        # Load data to get coordinates for graph building
        logger.info("Loading dataset for graph construction...")
        dataset = xr.open_dataset(args.data)
        lat_coords = dataset.lat.values
        lon_coords = dataset.lon.values
        logger.info(f"  Latitude: [{lat_coords.min():.2f}, {lat_coords.max():.2f}] "
                   f"({len(lat_coords)} points)")
        logger.info(f"  Longitude: [{lon_coords.min():.2f}, {lon_coords.max():.2f}] "
                   f"({len(lon_coords)} points)")
        logger.info("")
        
        # Build regional graph
        logger.info("Building regional graph...")
        graph_builder = RegionalGraphBuilder(
            region_config=region_config,
            lat_coords=lat_coords,
            lon_coords=lon_coords,
        )
        graph = graph_builder.build_graph()
        
        # Log graph statistics
        num_upstream = int(graph.nodes[types.UPSTREAM_NODE_TYPE].n_node[0])
        num_downstream = int(graph.nodes[types.DOWNSTREAM_NODE_TYPE].n_node[0])
        logger.info(f"  Upstream nodes: {num_upstream}")
        logger.info(f"  Downstream nodes: {num_downstream}")
        logger.info(f"  Total nodes: {num_upstream + num_downstream}")
        logger.info(f"  Graph structure: {graph}")
        logger.info("")
        
        # Initialize training pipeline
        logger.info("Initializing training pipeline...")
        pipeline = TrainingPipeline(
            model_config=model_config,
            region_config=region_config,
            training_config=training_config,
            data_path=args.data,
            graph=graph,
            output_dir=args.output_dir,
        )
        logger.info("")
        
        # Train model
        if args.resume_from:
            logger.info(f"Resuming training from: {args.resume_from}")
        else:
            logger.info("Starting training with GPU optimizations...")
        logger.info(f"  Data prefetching: {'enabled' if args.use_prefetch else 'disabled'}")
        if args.use_prefetch:
            logger.info(f"  Prefetch buffer size: {args.prefetch_buffer_size}")
        logger.info("=" * 80)
        params, normalizer = pipeline.train(
            train_end_year=args.train_end_year,
            val_end_year=args.val_end_year,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_start_date=args.test_start_date,
            seed=args.seed,
            use_prefetch=args.use_prefetch,
            prefetch_buffer_size=args.prefetch_buffer_size,
            resume_from=args.resume_from,
        )
        logger.info("=" * 80)
        logger.info("")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        logger.info("=" * 80)
        logger.info(f"Training completed successfully in {hours}h {minutes}m {seconds}s")
        logger.info(f"Checkpoints saved to: {args.output_dir}")
        logger.info(f"  - best_model.pkl: Best model based on validation loss")
        logger.info(f"  - normalizer.pkl: Data normalization statistics")
        logger.info(f"  - training.log: Training logs")
        logger.info("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration or data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
