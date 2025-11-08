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
"""Inference script for Regional Weather Prediction System.

This script generates precipitation predictions using a trained model.

Example usage:
    # Single-step prediction
    python scripts/run_inference.py \\
        --data data/processed/regional_weather.nc \\
        --checkpoint checkpoints/experiment_1/best_model.pkl \\
        --normalizer checkpoints/experiment_1/normalizer.pkl \\
        --target-time "2020-01-15 12:00:00" \\
        --output predictions/pred_20200115_12.nc
    
    # Multi-step prediction
    python scripts/run_inference.py \\
        --data data/processed/regional_weather.nc \\
        --checkpoint checkpoints/experiment_1/best_model.pkl \\
        --normalizer checkpoints/experiment_1/normalizer.pkl \\
        --initial-time "2020-01-15 00:00:00" \\
        --num-steps 10 \\
        --output predictions/pred_sequence.nc
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from graphcast_regional.config import ModelConfig, RegionConfig
from graphcast_regional.graph_builder import RegionalGraphBuilder
from graphcast_regional.inference import create_inference_pipeline
from graphcast_regional import types


def setup_logging(verbose: bool = False):
    """Configure logging for the script.
    
    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate precipitation predictions using trained model',
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
        '--output',
        type=str,
        required=True,
        help='Path to save predictions (NetCDF file)'
    )
    
    # Prediction mode
    prediction_mode = parser.add_mutually_exclusive_group(required=True)
    prediction_mode.add_argument(
        '--target-time',
        type=str,
        help='Target timestamp for single-step prediction (format: "YYYY-MM-DD HH:MM:SS")'
    )
    prediction_mode.add_argument(
        '--initial-time',
        type=str,
        help='Initial timestamp for multi-step prediction (format: "YYYY-MM-DD HH:MM:SS")'
    )
    
    parser.add_argument(
        '--num-steps',
        type=int,
        default=1,
        help='Number of 12-hour prediction steps (for multi-step mode)'
    )
    
    # Data windowing
    parser.add_argument(
        '--window-size',
        type=int,
        default=6,
        help='Number of historical timesteps to use as input (default: 6 for 3 days)'
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
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots of predictions'
    )
    parser.add_argument(
        '--viz-output',
        type=str,
        default=None,
        help='Directory to save visualization plots (default: same as output with _viz suffix)'
    )
    
    return parser.parse_args()


def create_visualizations(
    predictions: xr.Dataset, 
    dataset: xr.Dataset,
    output_dir: Path, 
    mode: str, 
    logger
):
    """Create visualization plots for predictions with ground truth comparison.
    
    Args:
        predictions: Dataset containing predictions
        dataset: Original dataset containing ground truth
        output_dir: Directory to save plots
        mode: Prediction mode ('single-step' or 'multi-step')
        logger: Logger instance
    """
    logger.info("Creating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    precip_pred = predictions["precipitation"]
    
    # Extract ground truth for the same times and crop to prediction region
    try:
        # Get the spatial extent of predictions (downstream region)
        pred_lats = precip_pred.lat.values
        pred_lons = precip_pred.lon.values
        
        # Crop ground truth to match prediction region
        precip_true = dataset["precipitation"].sel(
            time=precip_pred.time,
            lat=pred_lats,
            lon=pred_lons
        )
        has_ground_truth = True
        logger.info(f"  Ground truth cropped to downstream region: "
                   f"lat=[{pred_lats.min():.2f}, {pred_lats.max():.2f}], "
                   f"lon=[{pred_lons.min():.2f}, {pred_lons.max():.2f}]")
    except (KeyError, ValueError) as e:
        logger.warning(f"  Ground truth not available for prediction times/region: {e}")
        has_ground_truth = False
        precip_true = None
    
    if mode == "single-step":
        if has_ground_truth:
            # Side-by-side comparison
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            # Shared colorbar range
            vmin = min(float(precip_pred.min()), float(precip_true.min()))
            vmax = max(float(precip_pred.max()), float(precip_true.max()))
            
            # Ground truth
            precip_true.plot(
                ax=axes[0],
                cmap='YlGnBu',
                vmin=vmin,
                vmax=vmax,
                cbar_kwargs={'label': 'Precipitation (mm)'}
            )
            axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Longitude', fontsize=12)
            axes[0].set_ylabel('Latitude', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Prediction
            precip_pred.plot(
                ax=axes[1],
                cmap='YlGnBu',
                vmin=vmin,
                vmax=vmax,
                cbar_kwargs={'label': 'Precipitation (mm)'}
            )
            axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Longitude', fontsize=12)
            axes[1].set_ylabel('Latitude', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # Difference (Prediction - Ground Truth)
            diff = precip_pred - precip_true
            im = diff.plot(
                ax=axes[2],
                cmap='RdBu_r',
                center=0,
                cbar_kwargs={'label': 'Difference (mm)'}
            )
            axes[2].set_title('Difference (Pred - Truth)', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Longitude', fontsize=12)
            axes[2].set_ylabel('Latitude', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            # Add overall title with timestamp
            fig.suptitle(f'Precipitation Comparison - {precip_pred.time.values}', 
                        fontsize=16, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            plot_path = output_dir / 'comparison_map.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved comparison map: {plot_path}")
        else:
            # Single spatial map (prediction only)
            fig, ax = plt.subplots(figsize=(12, 8))
            
            precip_pred.plot(
                ax=ax,
                cmap='YlGnBu',
                cbar_kwargs={'label': 'Precipitation (mm)'}
            )
            ax.set_title(f'Predicted Precipitation\n{precip_pred.time.values}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = output_dir / 'prediction_map.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved spatial map: {plot_path}")
        
    else:  # multi-step
        num_steps = len(precip_pred.time)
        
        if has_ground_truth:
            # Create comparison grid: Ground Truth | Prediction | Difference
            ncols = 3
            nrows = num_steps
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows))
            if num_steps == 1:
                axes = axes.reshape(1, -1)
            
            # Shared colorbar range for predictions and ground truth
            vmin = min(float(precip_pred.min()), float(precip_true.min()))
            vmax = max(float(precip_pred.max()), float(precip_true.max()))
            
            for i, time_val in enumerate(precip_pred.time.values):
                # Ground truth
                precip_true.sel(time=time_val).plot(
                    ax=axes[i, 0],
                    cmap='YlGnBu',
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kwargs={'label': 'Precipitation (mm)'}
                )
                axes[i, 0].set_title(f'Ground Truth - Step {i+1}\n{pd.Timestamp(time_val).strftime("%Y-%m-%d %H:%M")}', 
                                    fontsize=11, fontweight='bold')
                axes[i, 0].set_xlabel('Longitude', fontsize=10)
                axes[i, 0].set_ylabel('Latitude', fontsize=10)
                axes[i, 0].grid(True, alpha=0.3)
                
                # Prediction
                precip_pred.sel(time=time_val).plot(
                    ax=axes[i, 1],
                    cmap='YlGnBu',
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kwargs={'label': 'Precipitation (mm)'}
                )
                axes[i, 1].set_title(f'Prediction - Step {i+1}\n{pd.Timestamp(time_val).strftime("%Y-%m-%d %H:%M")}', 
                                    fontsize=11, fontweight='bold')
                axes[i, 1].set_xlabel('Longitude', fontsize=10)
                axes[i, 1].set_ylabel('Latitude', fontsize=10)
                axes[i, 1].grid(True, alpha=0.3)
                
                # Difference
                diff = precip_pred.sel(time=time_val) - precip_true.sel(time=time_val)
                diff.plot(
                    ax=axes[i, 2],
                    cmap='RdBu_r',
                    center=0,
                    cbar_kwargs={'label': 'Difference (mm)'}
                )
                axes[i, 2].set_title(f'Difference - Step {i+1}\n{pd.Timestamp(time_val).strftime("%Y-%m-%d %H:%M")}', 
                                    fontsize=11, fontweight='bold')
                axes[i, 2].set_xlabel('Longitude', fontsize=10)
                axes[i, 2].set_ylabel('Latitude', fontsize=10)
                axes[i, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = output_dir / 'comparison_sequence.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved comparison sequence: {plot_path}")
        else:
            # Create multi-panel spatial maps (predictions only)
            ncols = min(3, num_steps)
            nrows = (num_steps + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
            if num_steps == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            vmin = float(precip_pred.min())
            vmax = float(precip_pred.max())
            
            for i, time_val in enumerate(precip_pred.time.values):
                ax = axes[i]
                precip_pred.sel(time=time_val).plot(
                    ax=ax,
                    cmap='YlGnBu',
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kwargs={'label': 'Precipitation (mm)'}
                )
                ax.set_title(f'Step {i+1}: {pd.Timestamp(time_val).strftime("%Y-%m-%d %H:%M")}', 
                            fontsize=11, fontweight='bold')
                ax.set_xlabel('Longitude', fontsize=10)
                ax.set_ylabel('Latitude', fontsize=10)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(num_steps, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plot_path = output_dir / 'prediction_sequence.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved sequence map: {plot_path}")
        
        # Time series of spatial statistics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        times = [pd.Timestamp(t) for t in precip_pred.time.values]
        mean_vals_pred = precip_pred.mean(dim=['lat', 'lon']).values
        max_vals_pred = precip_pred.max(dim=['lat', 'lon']).values
        min_vals_pred = precip_pred.min(dim=['lat', 'lon']).values
        std_vals_pred = precip_pred.std(dim=['lat', 'lon']).values
        
        if has_ground_truth:
            mean_vals_true = precip_true.mean(dim=['lat', 'lon']).values
            max_vals_true = precip_true.max(dim=['lat', 'lon']).values
            min_vals_true = precip_true.min(dim=['lat', 'lon']).values
            std_vals_true = precip_true.std(dim=['lat', 'lon']).values
        
        # Mean precipitation
        axes[0, 0].plot(times, mean_vals_pred, 'o-', linewidth=2, markersize=6, 
                       color='#2E86AB', label='Prediction')
        if has_ground_truth:
            axes[0, 0].plot(times, mean_vals_true, 's--', linewidth=2, markersize=6, 
                           color='#E63946', label='Ground Truth')
            axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_title('Mean Precipitation', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Precipitation (mm)', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Max precipitation
        axes[0, 1].plot(times, max_vals_pred, 'o-', linewidth=2, markersize=6, 
                       color='#A23B72', label='Prediction')
        if has_ground_truth:
            axes[0, 1].plot(times, max_vals_true, 's--', linewidth=2, markersize=6, 
                           color='#E63946', label='Ground Truth')
            axes[0, 1].legend(fontsize=10)
        axes[0, 1].set_title('Maximum Precipitation', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Precipitation (mm)', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Min precipitation
        axes[1, 0].plot(times, min_vals_pred, 'o-', linewidth=2, markersize=6, 
                       color='#F18F01', label='Prediction')
        if has_ground_truth:
            axes[1, 0].plot(times, min_vals_true, 's--', linewidth=2, markersize=6, 
                           color='#E63946', label='Ground Truth')
            axes[1, 0].legend(fontsize=10)
        axes[1, 0].set_title('Minimum Precipitation', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Precipitation (mm)', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Standard deviation
        axes[1, 1].plot(times, std_vals_pred, 'o-', linewidth=2, markersize=6, 
                       color='#6A994E', label='Prediction')
        if has_ground_truth:
            axes[1, 1].plot(times, std_vals_true, 's--', linewidth=2, markersize=6, 
                           color='#E63946', label='Ground Truth')
            axes[1, 1].legend(fontsize=10)
        axes[1, 1].set_title('Spatial Std Dev', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Precipitation (mm)', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = output_dir / 'prediction_timeseries.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved time series: {plot_path}")
    
    # Histogram of precipitation values
    fig, ax = plt.subplots(figsize=(10, 6))
    
    precip_flat_pred = precip_pred.values.flatten()
    ax.hist(precip_flat_pred, bins=50, color='#2E86AB', alpha=0.6, 
           edgecolor='black', label='Prediction')
    
    if has_ground_truth:
        precip_flat_true = precip_true.values.flatten()
        ax.hist(precip_flat_true, bins=50, color='#E63946', alpha=0.6, 
               edgecolor='black', label='Ground Truth')
        ax.legend(fontsize=11)
    
    ax.set_xlabel('Precipitation (mm)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Precipitation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Prediction:\n'
    stats_text += f'  Mean: {precip_flat_pred.mean():.2f} mm\n'
    stats_text += f'  Median: {np.median(precip_flat_pred):.2f} mm\n'
    stats_text += f'  Std: {precip_flat_pred.std():.2f} mm\n'
    stats_text += f'  Max: {precip_flat_pred.max():.2f} mm'
    
    if has_ground_truth:
        stats_text += f'\n\nGround Truth:\n'
        stats_text += f'  Mean: {precip_flat_true.mean():.2f} mm\n'
        stats_text += f'  Median: {np.median(precip_flat_true):.2f} mm\n'
        stats_text += f'  Std: {precip_flat_true.std():.2f} mm\n'
        stats_text += f'  Max: {precip_flat_true.max():.2f} mm'
    
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / 'prediction_histogram.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved histogram: {plot_path}")
    
    logger.info(f"All visualizations saved to: {output_dir}")


def main():
    """Main inference workflow."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Regional Weather Prediction - Inference")
    logger.info("=" * 80)
    
    # Determine prediction mode
    if args.target_time:
        mode = "single-step"
        target_time = pd.Timestamp(args.target_time)
        logger.info(f"Mode: Single-step prediction")
        logger.info(f"Target time: {target_time}")
    else:
        mode = "multi-step"
        initial_time = pd.Timestamp(args.initial_time)
        logger.info(f"Mode: Multi-step prediction")
        logger.info(f"Initial time: {initial_time}")
        logger.info(f"Number of steps: {args.num_steps}")
    
    logger.info(f"Data: {args.data}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Normalizer: {args.normalizer}")
    logger.info(f"Output: {args.output}")
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
        
        # Load data
        logger.info("Loading dataset...")
        dataset = xr.open_dataset(args.data)
        lat_coords = dataset.lat.values
        lon_coords = dataset.lon.values
        logger.info(f"  Time range: {dataset.time.values[0]} to {dataset.time.values[-1]}")
        logger.info(f"  Spatial extent: lat=[{lat_coords.min():.2f}, {lat_coords.max():.2f}], "
                   f"lon=[{lon_coords.min():.2f}, {lon_coords.max():.2f}]")
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
        
        # Generate predictions
        logger.info("Generating predictions...")
        logger.info("=" * 80)
        
        rng = jax.random.PRNGKey(args.seed)
        
        if mode == "single-step":
            predictions = pipeline.predict(
                data=dataset,
                target_time=target_time,
                rng=rng,
                window_size=args.window_size,
            )
        else:  # multi-step
            predictions = pipeline.predict_sequence(
                data=dataset,
                initial_time=initial_time,
                num_steps=args.num_steps,
                rng=rng,
                window_size=args.window_size,
            )
        
        logger.info("=" * 80)
        logger.info("")
        
        # Save predictions
        logger.info("Saving predictions...")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        predictions.attrs["description"] = "Regional precipitation predictions"
        predictions.attrs["model_checkpoint"] = args.checkpoint
        predictions.attrs["prediction_mode"] = mode
        if mode == "single-step":
            predictions.attrs["target_time"] = str(target_time)
        else:
            predictions.attrs["initial_time"] = str(initial_time)
            predictions.attrs["num_steps"] = args.num_steps
        
        # Save to NetCDF
        encoding = {
            "precipitation": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32"
            }
        }
        predictions.to_netcdf(args.output, encoding=encoding)
        
        # Log file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Predictions saved to {args.output} ({file_size_mb:.2f} MB)")
        logger.info("")
        
        # Log prediction statistics
        precip_values = predictions["precipitation"].values
        logger.info("Prediction statistics:")
        logger.info(f"  Min: {precip_values.min():.2f} mm")
        logger.info(f"  Max: {precip_values.max():.2f} mm")
        logger.info(f"  Mean: {precip_values.mean():.2f} mm")
        logger.info(f"  Std: {precip_values.std():.2f} mm")
        logger.info("")
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("")
            if args.viz_output:
                viz_dir = Path(args.viz_output)
            else:
                # Default: create directory next to output file
                output_path = Path(args.output)
                viz_dir = output_path.parent / f"{output_path.stem}_viz"
            
            create_visualizations(predictions, dataset, viz_dir, mode, logger)
            logger.info("")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"Inference completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Output saved to: {args.output}")
        if args.visualize:
            logger.info(f"Visualizations saved to: {viz_dir}")
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
