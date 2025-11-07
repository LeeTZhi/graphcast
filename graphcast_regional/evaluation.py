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
"""Evaluation and visualization tools for Regional Weather Prediction System."""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax.random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors
from matplotlib.patches import Rectangle

from graphcast_regional.config import RegionConfig
from graphcast_regional.inference import InferencePipeline


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.
    
    Attributes:
        mse_overall: Mean squared error across all samples.
        mse_light: MSE for light precipitation (0-10mm).
        mse_moderate: MSE for moderate precipitation (10-25mm).
        mse_heavy: MSE for heavy precipitation (>25mm).
        spatial_correlation: Spatial correlation coefficient.
        csi_1mm: Critical Success Index for 1mm threshold.
        csi_10mm: Critical Success Index for 10mm threshold.
        csi_25mm: Critical Success Index for 25mm threshold.
        num_samples: Number of samples evaluated.
    """
    mse_overall: float
    mse_light: float
    mse_moderate: float
    mse_heavy: float
    spatial_correlation: float
    csi_1mm: float
    csi_10mm: float
    csi_25mm: float
    num_samples: int
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Format metrics as readable string."""
        lines = [
            "Evaluation Metrics:",
            f"  Samples: {self.num_samples}",
            f"  MSE Overall: {self.mse_overall:.4f}",
            f"  MSE Light (0-10mm): {self.mse_light:.4f}",
            f"  MSE Moderate (10-25mm): {self.mse_moderate:.4f}",
            f"  MSE Heavy (>25mm): {self.mse_heavy:.4f}",
            f"  Spatial Correlation: {self.spatial_correlation:.4f}",
            f"  CSI @ 1mm: {self.csi_1mm:.4f}",
            f"  CSI @ 10mm: {self.csi_10mm:.4f}",
            f"  CSI @ 25mm: {self.csi_25mm:.4f}",
        ]
        return "\n".join(lines)


def compute_mse(predictions: xr.Dataset, targets: xr.Dataset) -> float:
    """Compute mean squared error.
    
    Args:
        predictions: Predicted precipitation values.
        targets: Actual precipitation values.
        
    Returns:
        Mean squared error.
    """
    pred_values = predictions["precipitation"].values
    target_values = targets["precipitation"].values
    
    mse = np.mean((pred_values - target_values) ** 2)
    return float(mse)


def compute_mse_by_intensity(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    intensity_ranges: List[Tuple[float, float]],
) -> List[float]:
    """Compute MSE for different precipitation intensity ranges.
    
    Args:
        predictions: Predicted precipitation values.
        targets: Actual precipitation values.
        intensity_ranges: List of (min, max) tuples defining intensity ranges.
            Use np.inf for unbounded upper range.
        
    Returns:
        List of MSE values for each intensity range.
    """
    pred_values = predictions["precipitation"].values.flatten()
    target_values = targets["precipitation"].values.flatten()
    
    mse_by_range = []
    
    for min_val, max_val in intensity_ranges:
        # Create mask for this intensity range based on target values
        mask = (target_values >= min_val) & (target_values < max_val)
        
        if np.sum(mask) == 0:
            # No samples in this range
            mse_by_range.append(np.nan)
        else:
            # Compute MSE for this range
            mse = np.mean((pred_values[mask] - target_values[mask]) ** 2)
            mse_by_range.append(float(mse))
    
    return mse_by_range


def compute_spatial_correlation(predictions: xr.Dataset, targets: xr.Dataset) -> float:
    """Compute spatial correlation coefficient.
    
    Computes the Pearson correlation coefficient between predicted and
    actual precipitation fields, averaged across all timesteps.
    
    Args:
        predictions: Predicted precipitation values.
        targets: Actual precipitation values.
        
    Returns:
        Spatial correlation coefficient.
    """
    pred_values = predictions["precipitation"].values
    target_values = targets["precipitation"].values
    
    # Handle different shapes
    if pred_values.ndim == 3:  # (time, lat, lon)
        # Compute correlation for each timestep and average
        correlations = []
        for t in range(pred_values.shape[0]):
            pred_flat = pred_values[t].flatten()
            target_flat = target_values[t].flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
            if np.sum(valid_mask) > 1:
                corr = np.corrcoef(pred_flat[valid_mask], target_flat[valid_mask])[0, 1]
                correlations.append(corr)
        
        if correlations:
            return float(np.mean(correlations))
        else:
            return np.nan
    
    elif pred_values.ndim == 2:  # (lat, lon)
        pred_flat = pred_values.flatten()
        target_flat = target_values.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        if np.sum(valid_mask) > 1:
            corr = np.corrcoef(pred_flat[valid_mask], target_flat[valid_mask])[0, 1]
            return float(corr)
        else:
            return np.nan
    
    else:
        raise ValueError(f"Unexpected prediction shape: {pred_values.shape}")


def compute_csi(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    threshold: float,
) -> float:
    """Compute Critical Success Index (CSI) for a precipitation threshold.
    
    CSI = hits / (hits + misses + false_alarms)
    
    Args:
        predictions: Predicted precipitation values.
        targets: Actual precipitation values.
        threshold: Precipitation threshold (mm).
        
    Returns:
        Critical Success Index.
    """
    pred_values = predictions["precipitation"].values.flatten()
    target_values = targets["precipitation"].values.flatten()
    
    # Binary classification: above or below threshold
    pred_binary = pred_values >= threshold
    target_binary = target_values >= threshold
    
    # Compute confusion matrix elements
    hits = np.sum(pred_binary & target_binary)
    misses = np.sum(~pred_binary & target_binary)
    false_alarms = np.sum(pred_binary & ~target_binary)
    
    # Compute CSI
    denominator = hits + misses + false_alarms
    
    if denominator == 0:
        # No precipitation events predicted or observed
        return np.nan
    
    csi = hits / denominator
    return float(csi)


def evaluate_predictions(
    predictions: xr.Dataset,
    targets: xr.Dataset,
) -> EvaluationMetrics:
    """Compute all evaluation metrics for predictions.
    
    Args:
        predictions: Predicted precipitation values.
        targets: Actual precipitation values.
        
    Returns:
        EvaluationMetrics object containing all computed metrics.
    """
    logger.info("Computing evaluation metrics")
    
    # Determine number of samples
    if "time" in predictions.dims:
        num_samples = len(predictions.time)
    else:
        num_samples = 1
    
    # Compute MSE overall
    mse_overall = compute_mse(predictions, targets)
    logger.info(f"MSE Overall: {mse_overall:.4f}")
    
    # Compute MSE by intensity ranges
    intensity_ranges = [
        (0.0, 10.0),      # Light
        (10.0, 25.0),     # Moderate
        (25.0, np.inf),   # Heavy
    ]
    mse_by_intensity = compute_mse_by_intensity(predictions, targets, intensity_ranges)
    mse_light, mse_moderate, mse_heavy = mse_by_intensity
    
    logger.info(f"MSE Light (0-10mm): {mse_light:.4f}")
    logger.info(f"MSE Moderate (10-25mm): {mse_moderate:.4f}")
    logger.info(f"MSE Heavy (>25mm): {mse_heavy:.4f}")
    
    # Compute spatial correlation
    spatial_corr = compute_spatial_correlation(predictions, targets)
    logger.info(f"Spatial Correlation: {spatial_corr:.4f}")
    
    # Compute CSI for different thresholds
    csi_1mm = compute_csi(predictions, targets, 1.0)
    csi_10mm = compute_csi(predictions, targets, 10.0)
    csi_25mm = compute_csi(predictions, targets, 25.0)
    
    logger.info(f"CSI @ 1mm: {csi_1mm:.4f}")
    logger.info(f"CSI @ 10mm: {csi_10mm:.4f}")
    logger.info(f"CSI @ 25mm: {csi_25mm:.4f}")
    
    # Create metrics object
    metrics = EvaluationMetrics(
        mse_overall=mse_overall,
        mse_light=mse_light,
        mse_moderate=mse_moderate,
        mse_heavy=mse_heavy,
        spatial_correlation=spatial_corr,
        csi_1mm=csi_1mm,
        csi_10mm=csi_10mm,
        csi_25mm=csi_25mm,
        num_samples=num_samples,
    )
    
    logger.info("Evaluation metrics computed successfully")
    
    return metrics


def save_evaluation_report(
    metrics: EvaluationMetrics,
    output_path: str,
):
    """Save evaluation metrics to JSON file.
    
    Args:
        metrics: EvaluationMetrics object.
        output_path: Path to save JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    logger.info(f"Evaluation report saved to {output_path}")



def plot_prediction_comparison(
    prediction: xr.Dataset,
    target: xr.Dataset,
    region_config: Optional[RegionConfig] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """Create side-by-side comparison plot of prediction vs actual.
    
    Args:
        prediction: Predicted precipitation (single timestep).
        target: Actual precipitation (single timestep).
        region_config: Optional region configuration for boundaries.
        title: Optional title for the plot.
        output_path: Optional path to save the figure.
        show_plot: Whether to display the plot.
    """
    # Extract values
    pred_values = prediction["precipitation"].values
    target_values = target["precipitation"].values
    
    # Handle time dimension if present
    if pred_values.ndim == 3:
        pred_values = pred_values[0]
        target_values = target_values[0]
    
    # Get coordinates
    lats = prediction["lat"].values
    lons = prediction["lon"].values
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define colormap and normalization for precipitation
    cmap = plt.cm.YlGnBu
    vmin = 0
    vmax = max(np.nanmax(pred_values), np.nanmax(target_values))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot 1: Prediction
    im1 = axes[0].pcolormesh(lons, lats, pred_values, cmap=cmap, norm=norm, shading='auto')
    axes[0].set_title("Prediction", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Longitude (°E)", fontsize=12)
    axes[0].set_ylabel("Latitude (°N)", fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im1, ax=axes[0], label="Precipitation (mm)")
    
    # Plot 2: Actual
    im2 = axes[1].pcolormesh(lons, lats, target_values, cmap=cmap, norm=norm, shading='auto')
    axes[1].set_title("Actual", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Longitude (°E)", fontsize=12)
    axes[1].set_ylabel("Latitude (°N)", fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im2, ax=axes[1], label="Precipitation (mm)")
    
    # Plot 3: Difference (Prediction - Actual)
    diff_values = pred_values - target_values
    diff_max = max(abs(np.nanmin(diff_values)), abs(np.nanmax(diff_values)))
    diff_norm = colors.Normalize(vmin=-diff_max, vmax=diff_max)
    
    im3 = axes[2].pcolormesh(lons, lats, diff_values, cmap='RdBu_r', norm=diff_norm, shading='auto')
    axes[2].set_title("Difference (Pred - Actual)", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Longitude (°E)", fontsize=12)
    axes[2].set_ylabel("Latitude (°N)", fontsize=12)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im3, ax=axes[2], label="Difference (mm)")
    
    # Add region boundaries if provided
    if region_config is not None:
        for ax in axes:
            # Downstream region boundary (solid line)
            rect_down = Rectangle(
                (region_config.downstream_lon_min, region_config.downstream_lat_min),
                region_config.downstream_lon_max - region_config.downstream_lon_min,
                region_config.downstream_lat_max - region_config.downstream_lat_min,
                linewidth=2, edgecolor='red', facecolor='none', linestyle='-',
                label='Downstream'
            )
            ax.add_patch(rect_down)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_time_series(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    lat: float,
    lon: float,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """Create time series plot for a specific location.
    
    Args:
        predictions: Predicted precipitation (multiple timesteps).
        targets: Actual precipitation (multiple timesteps).
        lat: Latitude of location.
        lon: Longitude of location.
        title: Optional title for the plot.
        output_path: Optional path to save the figure.
        show_plot: Whether to display the plot.
    """
    # Select nearest grid point
    pred_series = predictions["precipitation"].sel(lat=lat, lon=lon, method="nearest")
    target_series = targets["precipitation"].sel(lat=lat, lon=lon, method="nearest")
    
    # Get actual coordinates used
    actual_lat = float(pred_series.lat.values)
    actual_lon = float(pred_series.lon.values)
    
    # Get time values
    times = pd.DatetimeIndex(predictions.time.values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time series
    ax.plot(times, target_series.values, 'o-', label='Actual', linewidth=2, markersize=6)
    ax.plot(times, pred_series.values, 's-', label='Prediction', linewidth=2, markersize=6)
    
    # Formatting
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Precipitation (mm)", fontsize=12)
    ax.set_title(
        title or f"Time Series at ({actual_lat:.2f}°N, {actual_lon:.2f}°E)",
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Time series plot saved to {output_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_spatial_distribution(
    data: xr.Dataset,
    region_config: Optional[RegionConfig] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show_plot: bool = True,
):
    """Create spatial distribution plot with geographical boundaries.
    
    Args:
        data: Precipitation data (single timestep).
        region_config: Optional region configuration for boundaries.
        title: Optional title for the plot.
        output_path: Optional path to save the figure.
        show_plot: Whether to display the plot.
    """
    # Extract values
    values = data["precipitation"].values
    
    # Handle time dimension if present
    if values.ndim == 3:
        values = values[0]
    
    # Get coordinates
    lats = data["lat"].values
    lons = data["lon"].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colormap and normalization
    cmap = plt.cm.YlGnBu
    vmin = 0
    vmax = np.nanmax(values)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot precipitation
    im = ax.pcolormesh(lons, lats, values, cmap=cmap, norm=norm, shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Precipitation (mm)")
    
    # Add region boundaries if provided
    if region_config is not None:
        # Downstream region boundary (solid red line)
        rect_down = Rectangle(
            (region_config.downstream_lon_min, region_config.downstream_lat_min),
            region_config.downstream_lon_max - region_config.downstream_lon_min,
            region_config.downstream_lat_max - region_config.downstream_lat_min,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='-',
            label='Downstream Region'
        )
        ax.add_patch(rect_down)
        
        # Upstream region boundary (dashed blue line)
        rect_up = Rectangle(
            (region_config.upstream_lon_min, region_config.upstream_lat_min),
            region_config.upstream_lon_max - region_config.upstream_lon_min,
            region_config.upstream_lat_max - region_config.upstream_lat_min,
            linewidth=2, edgecolor='blue', facecolor='none', linestyle='--',
            label='Upstream Region'
        )
        ax.add_patch(rect_up)
        
        ax.legend(loc='upper right', fontsize=10)
    
    # Formatting
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title(title or "Precipitation Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Spatial distribution plot saved to {output_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_evaluation_visualizations(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    region_config: Optional[RegionConfig] = None,
    output_dir: Optional[str] = None,
    show_plots: bool = False,
):
    """Create comprehensive set of evaluation visualizations.
    
    Generates multiple plots for evaluating model predictions including
    comparison plots, time series, and spatial distributions.
    
    Args:
        predictions: Predicted precipitation (multiple timesteps).
        targets: Actual precipitation (multiple timesteps).
        region_config: Optional region configuration for boundaries.
        output_dir: Optional directory to save all plots.
        show_plots: Whether to display plots interactively.
    """
    logger.info("Creating evaluation visualizations")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get time dimension
    if "time" in predictions.dims:
        times = pd.DatetimeIndex(predictions.time.values)
        num_times = len(times)
    else:
        times = None
        num_times = 1
    
    # Create comparison plots for each timestep (limit to first 5 for brevity)
    max_plots = min(5, num_times)
    
    for i in range(max_plots):
        if times is not None:
            pred_t = predictions.isel(time=i)
            target_t = targets.isel(time=i)
            time_str = times[i].strftime("%Y-%m-%d %H:%M")
            title = f"Prediction Comparison - {time_str}"
            filename = f"comparison_{times[i].strftime('%Y%m%d_%H%M')}.png"
        else:
            pred_t = predictions
            target_t = targets
            title = "Prediction Comparison"
            filename = "comparison.png"
        
        output_path = output_dir / filename if output_dir else None
        
        plot_prediction_comparison(
            pred_t,
            target_t,
            region_config=region_config,
            title=title,
            output_path=output_path,
            show_plot=show_plots,
        )
    
    # Create time series plots for a few sample locations
    if times is not None and num_times > 1:
        # Get downstream region center
        if region_config:
            center_lat = (region_config.downstream_lat_min + region_config.downstream_lat_max) / 2
            center_lon = (region_config.downstream_lon_min + region_config.downstream_lon_max) / 2
        else:
            center_lat = float(predictions.lat.mean())
            center_lon = float(predictions.lon.mean())
        
        # Plot time series at center
        output_path = output_dir / "time_series_center.png" if output_dir else None
        plot_time_series(
            predictions,
            targets,
            lat=center_lat,
            lon=center_lon,
            title="Time Series at Region Center",
            output_path=output_path,
            show_plot=show_plots,
        )
    
    logger.info("Evaluation visualizations created successfully")



def evaluate_test_set(
    pipeline: InferencePipeline,
    test_data: xr.Dataset,
    test_times: List[pd.Timestamp],
    output_dir: str,
    create_visualizations: bool = True,
    rng: Optional[jax.random.PRNGKey] = None,
) -> EvaluationMetrics:
    """Evaluate model on test set and save results.
    
    Generates predictions for all test samples, computes evaluation metrics,
    and optionally creates visualizations.
    
    Args:
        pipeline: Initialized InferencePipeline.
        test_data: Test dataset containing atmospheric data.
        test_times: List of target timestamps for evaluation.
        output_dir: Directory to save evaluation results.
        create_visualizations: Whether to create visualization plots.
        rng: Optional random key for stochastic models.
        
    Returns:
        EvaluationMetrics object with computed metrics.
    """
    if rng is None:
        rng = jax.random.PRNGKey(42)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting batch evaluation on {len(test_times)} test samples")
    logger.info(f"Output directory: {output_dir}")
    
    # Generate predictions for all test times
    predictions_list = []
    targets_list = []
    
    for i, target_time in enumerate(test_times):
        logger.info(f"Processing test sample {i + 1}/{len(test_times)}: {target_time}")
        
        # Split random key
        rng, step_rng = jax.random.split(rng)
        
        try:
            # Generate prediction
            prediction = pipeline.predict(test_data, target_time, step_rng)
            
            # Extract target (actual precipitation at target_time)
            # Target should be in the downstream region only
            target = test_data.sel(time=target_time)
            
            # Extract downstream region from target
            downstream_lat_mask = (
                (target.lat >= pipeline.region_config.downstream_lat_min) &
                (target.lat <= pipeline.region_config.downstream_lat_max)
            )
            downstream_lon_mask = (
                (target.lon >= pipeline.region_config.downstream_lon_min) &
                (target.lon <= pipeline.region_config.downstream_lon_max)
            )
            
            target_downstream = target.sel(
                lat=target.lat[downstream_lat_mask],
                lon=target.lon[downstream_lon_mask]
            )
            
            # Store prediction and target
            predictions_list.append(prediction)
            targets_list.append(target_downstream)
            
        except Exception as e:
            logger.warning(f"Failed to process {target_time}: {e}")
            continue
    
    if not predictions_list:
        raise ValueError("No valid predictions generated for test set")
    
    logger.info(f"Successfully generated {len(predictions_list)} predictions")
    
    # Concatenate all predictions and targets
    predictions_combined = xr.concat(predictions_list, dim="time")
    targets_combined = xr.concat(targets_list, dim="time")
    
    # Save predictions to NetCDF
    predictions_path = output_dir / "predictions.nc"
    predictions_combined.to_netcdf(predictions_path)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # Compute evaluation metrics
    logger.info("Computing evaluation metrics")
    metrics = evaluate_predictions(predictions_combined, targets_combined)
    
    # Print metrics
    print("\n" + "=" * 60)
    print(metrics)
    print("=" * 60 + "\n")
    
    # Save metrics to JSON
    metrics_path = output_dir / "metrics.json"
    save_evaluation_report(metrics, metrics_path)
    
    # Save metrics as text file for easy reading
    metrics_text_path = output_dir / "metrics.txt"
    with open(metrics_text_path, 'w') as f:
        f.write(str(metrics))
    logger.info(f"Metrics saved to {metrics_text_path}")
    
    # Create visualizations if requested
    if create_visualizations:
        logger.info("Creating evaluation visualizations")
        viz_dir = output_dir / "visualizations"
        create_evaluation_visualizations(
            predictions_combined,
            targets_combined,
            region_config=pipeline.region_config,
            output_dir=viz_dir,
            show_plots=False,
        )
    
    logger.info("Batch evaluation completed successfully")
    
    return metrics


def load_test_split(
    data_path: str,
    test_start_time: str,
    test_end_time: Optional[str] = None,
) -> Tuple[xr.Dataset, List[pd.Timestamp]]:
    """Load test dataset and generate list of test timestamps.
    
    Args:
        data_path: Path to NetCDF dataset file.
        test_start_time: Start time for test period (ISO format string).
        test_end_time: Optional end time for test period (ISO format string).
            If None, uses all data from test_start_time onwards.
        
    Returns:
        Tuple of (test_data, test_times) where test_data is the xarray Dataset
        and test_times is a list of target timestamps for evaluation.
    """
    logger.info(f"Loading test data from {data_path}")
    
    # Load full dataset
    data = xr.open_dataset(data_path)
    
    # Parse time strings
    test_start = pd.Timestamp(test_start_time)
    test_end = pd.Timestamp(test_end_time) if test_end_time else None
    
    # Filter to test period
    if test_end:
        test_data = data.sel(time=slice(test_start, test_end))
    else:
        test_data = data.sel(time=slice(test_start, None))
    
    logger.info(f"Test period: {test_start} to {test_data.time.values[-1]}")
    logger.info(f"Test data shape: {test_data.dims}")
    
    # Generate list of target timestamps
    # We need at least 2 timesteps before each target (t-12h and t)
    # So target times start from the 3rd timestep onwards
    all_times = pd.DatetimeIndex(test_data.time.values)
    
    if len(all_times) < 3:
        raise ValueError(
            f"Test data must have at least 3 timesteps, got {len(all_times)}"
        )
    
    # Target times are all times from index 2 onwards
    # (we need t-12h and t as input, so target is t+12h)
    test_times = list(all_times[2:])
    
    logger.info(f"Generated {len(test_times)} test target times")
    
    return test_data, test_times


def run_full_evaluation(
    pipeline: InferencePipeline,
    data_path: str,
    test_start_time: str,
    output_dir: str,
    test_end_time: Optional[str] = None,
    create_visualizations: bool = True,
    rng: Optional[jax.random.PRNGKey] = None,
) -> EvaluationMetrics:
    """Run complete evaluation pipeline from data loading to results.
    
    Convenience function that loads test data, generates predictions,
    computes metrics, and creates visualizations.
    
    Args:
        pipeline: Initialized InferencePipeline.
        data_path: Path to NetCDF dataset file.
        test_start_time: Start time for test period (ISO format string).
        output_dir: Directory to save evaluation results.
        test_end_time: Optional end time for test period.
        create_visualizations: Whether to create visualization plots.
        rng: Optional random key for stochastic models.
        
    Returns:
        EvaluationMetrics object with computed metrics.
    """
    logger.info("Starting full evaluation pipeline")
    
    # Load test data
    test_data, test_times = load_test_split(
        data_path,
        test_start_time,
        test_end_time,
    )
    
    # Run evaluation
    metrics = evaluate_test_set(
        pipeline,
        test_data,
        test_times,
        output_dir,
        create_visualizations=create_visualizations,
        rng=rng,
    )
    
    logger.info("Full evaluation pipeline completed")
    
    return metrics
