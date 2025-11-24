"""Evaluation metrics for ConvLSTM weather prediction.

This module implements evaluation metrics for assessing precipitation prediction
accuracy in the downstream region. It provides RMSE, MAE, and CSI metrics that
can be computed for both experiments (baseline and upstream) on the same region
for fair comparison.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import xarray as xr


logger = logging.getLogger(__name__)


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
