"""Evaluation metrics for ConvLSTM weather prediction.

This module implements evaluation metrics for assessing precipitation prediction
accuracy in the downstream region. It provides RMSE, MAE, and CSI metrics that
can be computed for both experiments (baseline and upstream) on the same region
for fair comparison.

For multi-variable predictions, it also provides per-variable and per-timestep
metrics for comprehensive evaluation of atmospheric variables and rolling forecasts.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr


logger = logging.getLogger(__name__)


# Variable names for the 56 channels
# Channels 0-54: Atmospheric variables (5 vars × 11 levels)
# Channel 55: Precipitation
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 100]  # hPa
ATMOSPHERIC_VARS = ['DPT', 'GPH', 'TEM', 'U', 'V']  # 5 variables

def get_variable_names() -> List[str]:
    """Get list of all 56 variable names in channel order.
    
    Returns:
        List of variable names: ['DPT_1000', 'DPT_925', ..., 'V_100', 'precipitation']
    """
    var_names = []
    for var in ATMOSPHERIC_VARS:
        for level in PRESSURE_LEVELS:
            var_names.append(f"{var}_{level}")
    var_names.append("precipitation")
    return var_names


def compute_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute Root Mean Squared Error (RMSE) for downstream region.
    
    RMSE measures the average magnitude of prediction errors. It is sensitive
    to large errors and provides a measure in the same units as the predictions.
    
    Args:
        predictions: Predicted precipitation values [B, H, W] or [B, 1, H, W]
        targets: Ground truth precipitation values [B, H, W]
        mask: Optional boolean mask for downstream region [H, W] or [B, H, W]
              If None, computes RMSE over all grid points
    
    Returns:
        RMSE value (float)
        
    Examples:
        >>> pred = torch.randn(4, 50, 60)
        >>> target = torch.randn(4, 50, 60)
        >>> rmse = compute_rmse(pred, target)
        >>> rmse > 0
        True
    """
    # Handle predictions with channel dimension [B, 1, H, W]
    if predictions.dim() == 4 and predictions.size(1) == 1:
        predictions = predictions.squeeze(1)  # [B, H, W]
    
    # Ensure predictions and targets have the same shape
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )
    
    # Apply mask if provided
    if mask is not None:
        # Expand mask to match batch dimension if needed
        if mask.dim() == 2:  # [H, W]
            mask = mask.unsqueeze(0).expand_as(predictions)  # [B, H, W]
        
        # Apply mask
        predictions = predictions[mask]
        targets = targets[mask]
    else:
        # Flatten all dimensions
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    # Compute squared errors
    squared_errors = (predictions - targets) ** 2
    
    # Compute mean squared error
    mse = squared_errors.mean()
    
    # Compute RMSE
    rmse = torch.sqrt(mse)
    
    return float(rmse.item())


def compute_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute Mean Absolute Error (MAE) for downstream region.
    
    MAE measures the average absolute difference between predictions and targets.
    It is less sensitive to outliers than RMSE and provides a measure in the
    same units as the predictions.
    
    Args:
        predictions: Predicted precipitation values [B, H, W] or [B, 1, H, W]
        targets: Ground truth precipitation values [B, H, W]
        mask: Optional boolean mask for downstream region [H, W] or [B, H, W]
              If None, computes MAE over all grid points
    
    Returns:
        MAE value (float)
        
    Examples:
        >>> pred = torch.randn(4, 50, 60)
        >>> target = torch.randn(4, 50, 60)
        >>> mae = compute_mae(pred, target)
        >>> mae > 0
        True
    """
    # Handle predictions with channel dimension [B, 1, H, W]
    if predictions.dim() == 4 and predictions.size(1) == 1:
        predictions = predictions.squeeze(1)  # [B, H, W]
    
    # Ensure predictions and targets have the same shape
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )
    
    # Apply mask if provided
    if mask is not None:
        # Expand mask to match batch dimension if needed
        if mask.dim() == 2:  # [H, W]
            mask = mask.unsqueeze(0).expand_as(predictions)  # [B, H, W]
        
        # Apply mask
        predictions = predictions[mask]
        targets = targets[mask]
    else:
        # Flatten all dimensions
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    # Compute absolute errors
    absolute_errors = torch.abs(predictions - targets)
    
    # Compute mean absolute error
    mae = absolute_errors.mean()
    
    return float(mae.item())


def compute_csi(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    mask: Optional[torch.Tensor] = None
) -> float:
    """Compute Critical Success Index (CSI) for precipitation threshold.
    
    CSI measures the fraction of correctly predicted precipitation events
    (above a threshold) relative to all predicted and observed events.
    
    CSI = hits / (hits + misses + false_alarms)
    
    where:
    - hits: Both predicted and observed above threshold
    - misses: Observed above threshold but predicted below
    - false_alarms: Predicted above threshold but observed below
    
    CSI ranges from 0 (no skill) to 1 (perfect prediction).
    
    Args:
        predictions: Predicted precipitation values [B, H, W] or [B, 1, H, W]
        targets: Ground truth precipitation values [B, H, W]
        threshold: Precipitation threshold (mm) for event detection
        mask: Optional boolean mask for downstream region [H, W] or [B, H, W]
              If None, computes CSI over all grid points
    
    Returns:
        CSI value (float), or NaN if no events predicted or observed
        
    Examples:
        >>> pred = torch.rand(4, 50, 60) * 20  # 0-20mm
        >>> target = torch.rand(4, 50, 60) * 20
        >>> csi = compute_csi(pred, target, threshold=1.0)
        >>> 0 <= csi <= 1 or np.isnan(csi)
        True
    """
    # Handle predictions with channel dimension [B, 1, H, W]
    if predictions.dim() == 4 and predictions.size(1) == 1:
        predictions = predictions.squeeze(1)  # [B, H, W]
    
    # Ensure predictions and targets have the same shape
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )
    
    # Apply mask if provided
    if mask is not None:
        # Expand mask to match batch dimension if needed
        if mask.dim() == 2:  # [H, W]
            mask = mask.unsqueeze(0).expand_as(predictions)  # [B, H, W]
        
        # Apply mask
        predictions = predictions[mask]
        targets = targets[mask]
    else:
        # Flatten all dimensions
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    # Binary classification: above or below threshold
    pred_binary = predictions >= threshold
    target_binary = targets >= threshold
    
    # Compute confusion matrix elements
    hits = (pred_binary & target_binary).sum().item()
    misses = (~pred_binary & target_binary).sum().item()
    false_alarms = (pred_binary & ~target_binary).sum().item()
    
    # Compute CSI
    denominator = hits + misses + false_alarms
    
    if denominator == 0:
        # No precipitation events predicted or observed
        logger.warning(
            f"No precipitation events at threshold {threshold}mm. "
            f"Returning NaN for CSI."
        )
        return float('nan')
    
    csi = hits / denominator
    return float(csi)


def compute_metrics_for_region(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    csi_thresholds: Optional[list] = None
) -> Dict[str, float]:
    """Compute all evaluation metrics for a specific region.
    
    Computes RMSE, MAE, and CSI at multiple thresholds for the specified
    region (typically the downstream region for fair comparison).
    
    Args:
        predictions: Predicted precipitation values [B, H, W] or [B, 1, H, W]
        targets: Ground truth precipitation values [B, H, W]
        mask: Optional boolean mask for region [H, W] or [B, H, W]
              If None, computes metrics over all grid points
        csi_thresholds: List of precipitation thresholds (mm) for CSI computation
                       Default: [1.0, 5.0, 10.0, 25.0]
    
    Returns:
        Dictionary containing:
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'csi_Xmm': CSI at X mm threshold (for each threshold)
        
    Examples:
        >>> pred = torch.rand(4, 50, 60) * 20
        >>> target = torch.rand(4, 50, 60) * 20
        >>> metrics = compute_metrics_for_region(pred, target)
        >>> 'rmse' in metrics and 'mae' in metrics
        True
    """
    if csi_thresholds is None:
        csi_thresholds = [1.0, 5.0, 10.0, 25.0]
    
    logger.info("Computing evaluation metrics...")
    
    # Compute RMSE
    rmse = compute_rmse(predictions, targets, mask)
    logger.info(f"RMSE: {rmse:.4f} mm")
    
    # Compute MAE
    mae = compute_mae(predictions, targets, mask)
    logger.info(f"MAE: {mae:.4f} mm")
    
    # Compute CSI at different thresholds
    metrics = {
        'rmse': rmse,
        'mae': mae,
    }
    
    for threshold in csi_thresholds:
        csi = compute_csi(predictions, targets, threshold, mask)
        metric_name = f'csi_{int(threshold)}mm'
        metrics[metric_name] = csi
        logger.info(f"CSI @ {threshold}mm: {csi:.4f}")
    
    logger.info("Evaluation metrics computed successfully")
    
    return metrics


def create_downstream_mask(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    downstream_lat_min: float,
    downstream_lat_max: float,
    downstream_lon_min: float,
    downstream_lon_max: float
) -> torch.Tensor:
    """Create boolean mask for downstream region.
    
    Creates a spatial mask that selects only the downstream region grid points.
    This is used to ensure metrics are computed only on the downstream region
    for fair comparison between experiments.
    
    Args:
        lat_coords: Array of latitude coordinates [H]
        lon_coords: Array of longitude coordinates [W]
        downstream_lat_min: Minimum latitude for downstream region
        downstream_lat_max: Maximum latitude for downstream region
        downstream_lon_min: Minimum longitude for downstream region
        downstream_lon_max: Maximum longitude for downstream region
    
    Returns:
        Boolean mask tensor [H, W] where True indicates downstream region
        
    Examples:
        >>> lats = np.linspace(20, 50, 100)
        >>> lons = np.linspace(70, 130, 150)
        >>> mask = create_downstream_mask(lats, lons, 25, 40, 110, 125)
        >>> mask.shape
        torch.Size([100, 150])
    """
    # Create 2D coordinate grids
    lat_grid, lon_grid = np.meshgrid(lat_coords, lon_coords, indexing='ij')
    
    # Create boolean mask for downstream region
    lat_mask = (lat_grid >= downstream_lat_min) & (lat_grid <= downstream_lat_max)
    lon_mask = (lon_grid >= downstream_lon_min) & (lon_grid <= downstream_lon_max)
    
    # Combine masks
    downstream_mask = lat_mask & lon_mask
    
    # Convert to torch tensor
    mask_tensor = torch.from_numpy(downstream_mask)
    
    return mask_tensor


def compare_experiments(
    exp1_predictions: torch.Tensor,
    exp2_predictions: torch.Tensor,
    targets: torch.Tensor,
    downstream_mask: torch.Tensor,
    exp1_name: str = "Experiment 1 (Baseline)",
    exp2_name: str = "Experiment 2 (Upstream)",
    csi_thresholds: Optional[list] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compare metrics for two experiments on the same downstream region.
    
    Computes evaluation metrics for both experiments using the same downstream
    region mask to ensure fair comparison. This is the primary function for
    comparing the baseline experiment (downstream only) with the upstream
    experiment (downstream + upstream).
    
    Args:
        exp1_predictions: Predictions from experiment 1 [B, H, W] or [B, 1, H, W]
        exp2_predictions: Predictions from experiment 2 [B, H, W] or [B, 1, H, W]
        targets: Ground truth precipitation [B, H, W]
        downstream_mask: Boolean mask for downstream region [H, W]
        exp1_name: Name for experiment 1 (default: "Experiment 1 (Baseline)")
        exp2_name: Name for experiment 2 (default: "Experiment 2 (Upstream)")
        csi_thresholds: List of precipitation thresholds for CSI
                       Default: [1.0, 5.0, 10.0, 25.0]
    
    Returns:
        Tuple of (exp1_metrics, exp2_metrics) dictionaries
        
    Examples:
        >>> pred1 = torch.rand(4, 50, 60) * 20
        >>> pred2 = torch.rand(4, 50, 60) * 20
        >>> target = torch.rand(4, 50, 60) * 20
        >>> mask = torch.ones(50, 60, dtype=torch.bool)
        >>> m1, m2 = compare_experiments(pred1, pred2, target, mask)
        >>> 'rmse' in m1 and 'rmse' in m2
        True
    """
    logger.info("=" * 60)
    logger.info("Comparing experiments on downstream region")
    logger.info("=" * 60)
    
    # Compute metrics for experiment 1
    logger.info(f"\n{exp1_name}:")
    exp1_metrics = compute_metrics_for_region(
        exp1_predictions,
        targets,
        mask=downstream_mask,
        csi_thresholds=csi_thresholds
    )
    
    # Compute metrics for experiment 2
    logger.info(f"\n{exp2_name}:")
    exp2_metrics = compute_metrics_for_region(
        exp2_predictions,
        targets,
        mask=downstream_mask,
        csi_thresholds=csi_thresholds
    )
    
    # Print comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary:")
    logger.info("=" * 60)
    
    logger.info(f"\nRMSE:")
    logger.info(f"  {exp1_name}: {exp1_metrics['rmse']:.4f} mm")
    logger.info(f"  {exp2_name}: {exp2_metrics['rmse']:.4f} mm")
    rmse_improvement = ((exp1_metrics['rmse'] - exp2_metrics['rmse']) / 
                        exp1_metrics['rmse'] * 100)
    logger.info(f"  Improvement: {rmse_improvement:+.2f}%")
    
    logger.info(f"\nMAE:")
    logger.info(f"  {exp1_name}: {exp1_metrics['mae']:.4f} mm")
    logger.info(f"  {exp2_name}: {exp2_metrics['mae']:.4f} mm")
    mae_improvement = ((exp1_metrics['mae'] - exp2_metrics['mae']) / 
                       exp1_metrics['mae'] * 100)
    logger.info(f"  Improvement: {mae_improvement:+.2f}%")
    
    # Print CSI comparisons
    if csi_thresholds is None:
        csi_thresholds = [1.0, 5.0, 10.0, 25.0]
    
    for threshold in csi_thresholds:
        metric_name = f'csi_{int(threshold)}mm'
        if metric_name in exp1_metrics and metric_name in exp2_metrics:
            logger.info(f"\nCSI @ {threshold}mm:")
            logger.info(f"  {exp1_name}: {exp1_metrics[metric_name]:.4f}")
            logger.info(f"  {exp2_name}: {exp2_metrics[metric_name]:.4f}")
            
            # CSI improvement (higher is better)
            if not np.isnan(exp1_metrics[metric_name]) and not np.isnan(exp2_metrics[metric_name]):
                csi_improvement = ((exp2_metrics[metric_name] - exp1_metrics[metric_name]) / 
                                  exp1_metrics[metric_name] * 100)
                logger.info(f"  Improvement: {csi_improvement:+.2f}%")
    
    logger.info("=" * 60)
    
    return exp1_metrics, exp2_metrics


def compute_metrics_from_xarray(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    downstream_lat_min: float,
    downstream_lat_max: float,
    downstream_lon_min: float,
    downstream_lon_max: float,
    csi_thresholds: Optional[list] = None
) -> Dict[str, float]:
    """Compute metrics from xarray Datasets (convenience function).
    
    This is a convenience wrapper that handles xarray Dataset inputs and
    automatically extracts the downstream region for metric computation.
    
    Args:
        predictions: Predicted precipitation as xarray Dataset
        targets: Ground truth precipitation as xarray Dataset
        downstream_lat_min: Minimum latitude for downstream region
        downstream_lat_max: Maximum latitude for downstream region
        downstream_lon_min: Minimum longitude for downstream region
        downstream_lon_max: Maximum longitude for downstream region
        csi_thresholds: List of precipitation thresholds for CSI
    
    Returns:
        Dictionary of evaluation metrics
        
    Examples:
        >>> # Create sample xarray datasets
        >>> import xarray as xr
        >>> pred_ds = xr.Dataset({
        ...     'precipitation': (['time', 'lat', 'lon'], np.random.rand(10, 50, 60))
        ... })
        >>> target_ds = xr.Dataset({
        ...     'precipitation': (['time', 'lat', 'lon'], np.random.rand(10, 50, 60))
        ... })
        >>> metrics = compute_metrics_from_xarray(
        ...     pred_ds, target_ds, 25, 40, 110, 125
        ... )
    """
    # Extract precipitation arrays
    pred_array = predictions["precipitation"].values
    target_array = targets["precipitation"].values
    
    # Get coordinates
    lat_coords = predictions["lat"].values
    lon_coords = predictions["lon"].values
    
    # Convert to torch tensors
    pred_tensor = torch.from_numpy(pred_array).float()
    target_tensor = torch.from_numpy(target_array).float()
    
    # Create downstream mask
    downstream_mask = create_downstream_mask(
        lat_coords,
        lon_coords,
        downstream_lat_min,
        downstream_lat_max,
        downstream_lon_min,
        downstream_lon_max
    )
    
    # Compute metrics
    metrics = compute_metrics_for_region(
        pred_tensor,
        target_tensor,
        mask=downstream_mask,
        csi_thresholds=csi_thresholds
    )
    
    return metrics



def compute_per_variable_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, Dict[str, float]]:
    """Compute RMSE and MAE for each variable separately in multi-variable predictions.
    
    This function computes per-variable metrics for all 56 channels:
    - 55 atmospheric variables (5 vars × 11 pressure levels)
    - 1 precipitation variable
    
    Args:
        predictions: Predicted values [B, 56, H, W]
        targets: Ground truth values [B, 56, H, W]
        mask: Optional boolean mask for region [H, W] or [B, H, W]
    
    Returns:
        Dictionary mapping variable names to their metrics:
        {
            'DPT_1000': {'rmse': 1.23, 'mae': 0.98},
            'DPT_925': {'rmse': 1.45, 'mae': 1.12},
            ...
            'precipitation': {'rmse': 2.34, 'mae': 1.87}
        }
    
    Raises:
        ValueError: If predictions don't have 56 channels
    """
    if predictions.dim() != 4 or predictions.size(1) != 56:
        raise ValueError(
            f"Expected predictions shape [B, 56, H, W], got {predictions.shape}"
        )
    
    if targets.dim() != 4 or targets.size(1) != 56:
        raise ValueError(
            f"Expected targets shape [B, 56, H, W], got {targets.shape}"
        )
    
    var_names = get_variable_names()
    metrics = {}
    
    logger.info("Computing per-variable metrics for 56 channels...")
    
    for channel_idx, var_name in enumerate(var_names):
        # Extract single channel
        pred_channel = predictions[:, channel_idx:channel_idx+1, :, :]  # [B, 1, H, W]
        target_channel = targets[:, channel_idx, :, :]  # [B, H, W]
        
        # Compute metrics for this channel
        rmse = compute_rmse(pred_channel, target_channel, mask)
        mae = compute_mae(pred_channel, target_channel, mask)
        
        metrics[var_name] = {
            'rmse': rmse,
            'mae': mae
        }
    
    logger.info(f"Computed metrics for {len(var_names)} variables")
    
    return metrics


def compute_per_timestep_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[int, Dict[str, float]]:
    """Compute RMSE and MAE for each timestep in rolling forecast predictions.
    
    This function computes metrics separately for each predicted timestep
    in a rolling forecast sequence.
    
    Args:
        predictions: Predicted values [B, T, C, H, W] where T is number of timesteps
        targets: Ground truth values [B, T, C, H, W]
        mask: Optional boolean mask for region [H, W] or [B, H, W]
    
    Returns:
        Dictionary mapping timestep to metrics:
        {
            1: {'rmse': 1.23, 'mae': 0.98},
            2: {'rmse': 1.45, 'mae': 1.12},
            ...
        }
    
    Raises:
        ValueError: If predictions don't have timestep dimension
    """
    if predictions.dim() != 5:
        raise ValueError(
            f"Expected predictions shape [B, T, C, H, W], got {predictions.shape}"
        )
    
    if targets.dim() != 5:
        raise ValueError(
            f"Expected targets shape [B, T, C, H, W], got {targets.shape}"
        )
    
    num_timesteps = predictions.size(1)
    num_channels = predictions.size(2)
    
    logger.info(f"Computing per-timestep metrics for {num_timesteps} timesteps...")
    
    metrics = {}
    
    for t in range(num_timesteps):
        # Extract timestep
        pred_t = predictions[:, t, :, :, :]  # [B, C, H, W]
        target_t = targets[:, t, :, :, :]  # [B, C, H, W]
        
        # For multi-channel predictions, compute overall metrics
        # (average across all channels)
        if num_channels == 1:
            # Single-variable mode (precipitation only)
            rmse = compute_rmse(pred_t, target_t.squeeze(1), mask)
            mae = compute_mae(pred_t, target_t.squeeze(1), mask)
        else:
            # Multi-variable mode: compute metrics for all channels
            # Flatten channel dimension for overall metric
            pred_flat = pred_t.reshape(pred_t.size(0), -1, pred_t.size(-2), pred_t.size(-1))
            target_flat = target_t.reshape(target_t.size(0), -1, target_t.size(-2), target_t.size(-1))
            
            # Compute average RMSE and MAE across all channels
            rmse_sum = 0.0
            mae_sum = 0.0
            for c in range(num_channels):
                rmse_sum += compute_rmse(pred_t[:, c:c+1, :, :], target_t[:, c, :, :], mask)
                mae_sum += compute_mae(pred_t[:, c:c+1, :, :], target_t[:, c, :, :], mask)
            
            rmse = rmse_sum / num_channels
            mae = mae_sum / num_channels
        
        metrics[t + 1] = {  # 1-indexed timesteps
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"Timestep {t+1}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return metrics


def compute_weighted_aggregated_score(
    per_variable_metrics: Dict[str, Dict[str, float]],
    precip_weight: float = 10.0
) -> Dict[str, float]:
    """Compute weighted aggregated scores across all variables.
    
    Computes overall RMSE and MAE by taking a weighted average where
    precipitation receives higher weight than atmospheric variables.
    
    Args:
        per_variable_metrics: Dictionary of per-variable metrics from compute_per_variable_metrics
        precip_weight: Weight multiplier for precipitation (default: 10.0)
    
    Returns:
        Dictionary with aggregated metrics:
        {
            'weighted_rmse': 2.34,
            'weighted_mae': 1.87,
            'precip_rmse': 3.45,
            'precip_mae': 2.67,
            'atmos_rmse': 1.23,
            'atmos_mae': 0.98
        }
    """
    # Separate precipitation and atmospheric metrics
    precip_metrics = per_variable_metrics.get('precipitation', {})
    
    # Compute average atmospheric metrics
    atmos_rmse_sum = 0.0
    atmos_mae_sum = 0.0
    atmos_count = 0
    
    for var_name, metrics in per_variable_metrics.items():
        if var_name != 'precipitation':
            atmos_rmse_sum += metrics['rmse']
            atmos_mae_sum += metrics['mae']
            atmos_count += 1
    
    atmos_rmse = atmos_rmse_sum / atmos_count if atmos_count > 0 else 0.0
    atmos_mae = atmos_mae_sum / atmos_count if atmos_count > 0 else 0.0
    
    # Compute weighted average
    # weighted_metric = (precip_weight * precip_metric + atmos_metric) / (precip_weight + 1)
    precip_rmse = precip_metrics.get('rmse', 0.0)
    precip_mae = precip_metrics.get('mae', 0.0)
    
    weighted_rmse = (precip_weight * precip_rmse + atmos_rmse) / (precip_weight + 1)
    weighted_mae = (precip_weight * precip_mae + atmos_mae) / (precip_weight + 1)
    
    logger.info("Computed weighted aggregated scores:")
    logger.info(f"  Weighted RMSE: {weighted_rmse:.4f}")
    logger.info(f"  Weighted MAE: {weighted_mae:.4f}")
    logger.info(f"  Precipitation RMSE: {precip_rmse:.4f}")
    logger.info(f"  Precipitation MAE: {precip_mae:.4f}")
    logger.info(f"  Atmospheric RMSE: {atmos_rmse:.4f}")
    logger.info(f"  Atmospheric MAE: {atmos_mae:.4f}")
    
    return {
        'weighted_rmse': weighted_rmse,
        'weighted_mae': weighted_mae,
        'precip_rmse': precip_rmse,
        'precip_mae': precip_mae,
        'atmos_rmse': atmos_rmse,
        'atmos_mae': atmos_mae
    }


def evaluate_multi_variable_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    precip_weight: float = 10.0,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, any]:
    """Comprehensive evaluation for multi-variable predictions.
    
    Computes:
    1. Per-variable RMSE and MAE for all 56 channels
    2. Weighted aggregated scores
    3. Exports results to CSV/JSON if output_dir is provided
    
    Args:
        predictions: Predicted values [B, 56, H, W]
        targets: Ground truth values [B, 56, H, W]
        mask: Optional boolean mask for region [H, W] or [B, H, W]
        precip_weight: Weight for precipitation in aggregated scores (default: 10.0)
        output_dir: Optional directory to save results as CSV/JSON
    
    Returns:
        Dictionary containing:
        {
            'per_variable': {...},  # Per-variable metrics
            'aggregated': {...}     # Weighted aggregated scores
        }
    """
    logger.info("=" * 60)
    logger.info("Multi-Variable Evaluation")
    logger.info("=" * 60)
    
    # Compute per-variable metrics
    per_variable_metrics = compute_per_variable_metrics(predictions, targets, mask)
    
    # Compute weighted aggregated scores
    aggregated_metrics = compute_weighted_aggregated_score(per_variable_metrics, precip_weight)
    
    results = {
        'per_variable': per_variable_metrics,
        'aggregated': aggregated_metrics
    }
    
    # Export results if output directory is provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        csv_path = output_path / "multi_variable_metrics.csv"
        export_metrics_to_csv(per_variable_metrics, csv_path)
        logger.info(f"Exported per-variable metrics to {csv_path}")
        
        # Export to JSON
        json_path = output_path / "multi_variable_metrics.json"
        export_metrics_to_json(results, json_path)
        logger.info(f"Exported all metrics to {json_path}")
    
    logger.info("=" * 60)
    
    return results


def evaluate_rolling_forecast(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    precip_weight: float = 10.0,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, any]:
    """Comprehensive evaluation for rolling forecast predictions.
    
    Computes:
    1. Per-timestep metrics for each step in the rolling forecast
    2. Per-variable metrics at each timestep (if multi-variable)
    3. Exports results to CSV/JSON if output_dir is provided
    
    Args:
        predictions: Predicted values [B, T, C, H, W] where T is number of timesteps
        targets: Ground truth values [B, T, C, H, W]
        mask: Optional boolean mask for region [H, W] or [B, H, W]
        precip_weight: Weight for precipitation in aggregated scores (default: 10.0)
        output_dir: Optional directory to save results as CSV/JSON
    
    Returns:
        Dictionary containing:
        {
            'per_timestep': {...},           # Overall metrics per timestep
            'per_timestep_per_variable': {...}  # Per-variable metrics at each timestep (if multi-variable)
        }
    """
    logger.info("=" * 60)
    logger.info("Rolling Forecast Evaluation")
    logger.info("=" * 60)
    
    # Compute per-timestep metrics
    per_timestep_metrics = compute_per_timestep_metrics(predictions, targets, mask)
    
    results = {
        'per_timestep': per_timestep_metrics
    }
    
    # If multi-variable, also compute per-variable metrics at each timestep
    num_channels = predictions.size(2)
    if num_channels == 56:
        logger.info("\nComputing per-variable metrics at each timestep...")
        per_timestep_per_variable = {}
        
        for t in range(predictions.size(1)):
            pred_t = predictions[:, t, :, :, :]  # [B, 56, H, W]
            target_t = targets[:, t, :, :, :]  # [B, 56, H, W]
            
            var_metrics = compute_per_variable_metrics(pred_t, target_t, mask)
            per_timestep_per_variable[t + 1] = var_metrics
        
        results['per_timestep_per_variable'] = per_timestep_per_variable
    
    # Export results if output directory is provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export per-timestep metrics to CSV
        csv_path = output_path / "rolling_forecast_metrics.csv"
        export_timestep_metrics_to_csv(per_timestep_metrics, csv_path)
        logger.info(f"Exported per-timestep metrics to {csv_path}")
        
        # Export all results to JSON
        json_path = output_path / "rolling_forecast_metrics.json"
        export_metrics_to_json(results, json_path)
        logger.info(f"Exported all metrics to {json_path}")
    
    logger.info("=" * 60)
    
    return results


def export_metrics_to_csv(
    per_variable_metrics: Dict[str, Dict[str, float]],
    output_path: Union[str, Path]
) -> None:
    """Export per-variable metrics to CSV file.
    
    Args:
        per_variable_metrics: Dictionary of per-variable metrics
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Variable', 'RMSE', 'MAE'])
        
        # Write data
        for var_name, metrics in per_variable_metrics.items():
            writer.writerow([
                var_name,
                f"{metrics['rmse']:.6f}",
                f"{metrics['mae']:.6f}"
            ])


def export_timestep_metrics_to_csv(
    per_timestep_metrics: Dict[int, Dict[str, float]],
    output_path: Union[str, Path]
) -> None:
    """Export per-timestep metrics to CSV file.
    
    Args:
        per_timestep_metrics: Dictionary of per-timestep metrics
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Timestep', 'RMSE', 'MAE'])
        
        # Write data
        for timestep, metrics in sorted(per_timestep_metrics.items()):
            writer.writerow([
                timestep,
                f"{metrics['rmse']:.6f}",
                f"{metrics['mae']:.6f}"
            ])


def export_metrics_to_json(
    metrics: Dict,
    output_path: Union[str, Path]
) -> None:
    """Export metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as jsonfile:
        json.dump(metrics, jsonfile, indent=2)
