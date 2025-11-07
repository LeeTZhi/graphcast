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
"""Evaluation script for Regional Weather Prediction System.

This script evaluates a trained model on the test set, computes metrics,
and generates visualizations.

Example usage:
    python scripts/evaluate_model.py \\
        --data data/processed/regional_weather.nc \\
        --checkpoint checkpoints/experiment_1/best_model.pkl \\
        --normalizer checkpoints/experiment_1/normalizer.pkl \\
        --test-start-time "2020-01-01 00:00:00" \\
        --test-end-time "2020-12-31 23:59:59" \\
        --output-dir evaluation/experiment_1 \\
        --visualizations
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import xarray as xr

from graphcast_regional.config import ModelConfig, RegionConfig
from graphcast_regional.graph_builder import RegionalGraphBuilder
from graphcast_regional.inference import create_inference_pipeline
from graphcast_regional.evaluation import run_full_evaluation
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
    log_file = Path(output_dir) / "evaluation.log"
    
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
        description='Evaluate Regional Weather Prediction model on test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required paths
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to preprocessed NetCDF dataset'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pkl file)'
    )
    parser.add_argument(
        '--normalizer',
        type=str,
        required=True,
        help='Path to normalizer file (.pkl file)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save evaluation results'
    )
    
    # Test period
    parser.add_argument(
        '--test-start-time',
        type=str,
        required=True,
        help='Start time for test period (format: "YYYY-MM-DD HH:MM:SS")'
    )
    parser.add_argument(
        '--test-end-time',
        type=str,
        default=None,
        help='End time for test period (format: "YYYY-MM-DD HH:MM:SS"). '
             'If not specified, uses all data from test-start-time onwards.'
    )
    
    # Region configuration (should match training)
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
    
    # Model configuration (should match training)
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
    
    # Evaluation options
    parser.add_argument(
        '--visualizations',
        action='store_true',
        help='Create visualization plots (comparison plots, time series, etc.)'
    )
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
    """Main evaluation workflow."""
    args = parse_args()
    setup_logging(args.output_dir, args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Regional Weather Prediction Model Evaluation")
    logger.info("=" * 80)
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Normalizer: {args.normalizer}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Test period: {args.test_start_time} to {args.test_end_time or 'end'}")
    logger.info(f"  Create visualizations: {args.visualizations}")
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
        num_upstream = int(graph.nodes[types.UPSTREAM_NODE_TYPE].n_node[0])
        num_downstream = int(graph.nodes[types.DOWNSTREAM_NODE_TYPE].n_node[0])
        logger.info(f"  Upstream nodes: {num_upstream}")
        logger.info(f"  Downstream nodes: {num_downstream}")
        logger.info("")
        
        # Create inference pipeline
        logger.info("Creating inference pipeline...")
        pipeline = create_inference_pipeline(
            model_config=model_config,
            region_config=region_config,
            checkpoint_path=args.checkpoint,
            normalizer_path=args.normalizer,
            graph=graph,
        )
        logger.info("")
        
        # Run evaluation
        logger.info("Starting evaluation on test set...")
        logger.info("=" * 80)
        
        rng = jax.random.PRNGKey(args.seed)
        
        metrics = run_full_evaluation(
            pipeline=pipeline,
            data_path=args.data,
            test_start_time=args.test_start_time,
            output_dir=args.output_dir,
            test_end_time=args.test_end_time,
            create_visualizations=args.visualizations,
            rng=rng,
        )
        
        logger.info("=" * 80)
        logger.info("")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        logger.info("=" * 80)
        logger.info(f"Evaluation completed successfully in {hours}h {minutes}m {seconds}s")
        logger.info("")
        logger.info("Results saved to:")
        logger.info(f"  - {args.output_dir}/metrics.json: Evaluation metrics (JSON)")
        logger.info(f"  - {args.output_dir}/metrics.txt: Evaluation metrics (text)")
        logger.info(f"  - {args.output_dir}/predictions.nc: Model predictions (NetCDF)")
        logger.info(f"  - {args.output_dir}/evaluation.log: Evaluation logs")
        
        if args.visualizations:
            logger.info(f"  - {args.output_dir}/visualizations/: Visualization plots")
        
        logger.info("")
        logger.info("Summary of metrics:")
        logger.info(f"  MSE Overall: {metrics.mse_overall:.4f}")
        logger.info(f"  Spatial Correlation: {metrics.spatial_correlation:.4f}")
        logger.info(f"  CSI @ 10mm: {metrics.csi_10mm:.4f}")
        logger.info("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data or configuration: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
