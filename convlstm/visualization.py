"""Visualization utilities for ConvLSTM weather prediction.

This module provides functions for creating precipitation maps, comparison plots,
error/difference maps, and saving visualizations with descriptive filenames.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr


logger = logging.getLogger(__name__)


def plot_precipitation_map(
    precipitation: Union[xr.DataArray, np.ndarray],
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    title: str = "Precipitation",
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    colorbar_label: str = "Precipitation (mm)"
) -> plt.Axes:
    """Plot a precipitation map with colorbar.
    
    Creates a 2D map visualization of precipitation data with proper geographic
    coordinates and a colorbar. Uses a blue colormap by default, with white
    representing no precipitation and darker blue representing heavy rain.
    
    Args:
        precipitation: 2D array of precipitation values [H, W] or xarray DataArray.
                      If xarray, lat/lon coordinates are extracted automatically.
        lat_coords: Latitude coordinates (required if precipitation is numpy array).
        lon_coords: Longitude coordinates (required if precipitation is numpy array).
        title: Title for the plot.
        cmap: Matplotlib colormap name (default: 'Blues').
        vmin: Minimum value for colormap (default: 0.0).
        vmax: Maximum value for colormap (default: auto-scale to data max).
        ax: Matplotlib axes to plot on. If None, creates new figure.
        show_colorbar: Whether to show colorbar (default: True).
        colorbar_label: Label for colorbar (default: 'Precipitation (mm)').
    
    Returns:
        Matplotlib axes with the plot.
    
    Raises:
        ValueError: If precipitation is numpy array but lat/lon coords not provided.
    
    Example:
        >>> # Plot from xarray DataArray
        >>> ax = plot_precipitation_map(
        ...     prediction.precipitation,
        ...     title="Predicted Precipitation"
        ... )
        >>> plt.savefig('prediction.png')
        >>> plt.close()
        
        >>> # Plot from numpy array
        >>> ax = plot_precipitation_map(
        ...     pred_array,
        ...     lat_coords=lats,
        ...     lon_coords=lons,
        ...     title="Prediction"
        ... )
    """
    # Extract data and coordinates
    if isinstance(precipitation, xr.DataArray):
        data = precipitation.values
        if lat_coords is None and 'lat' in precipitation.dims:
            lat_coords = precipitation.lat.values
        if lon_coords is None and 'lon' in precipitation.dims:
            lon_coords = precipitation.lon.values
    else:
        data = precipitation
    
    # Validate coordinates
    if lat_coords is None or lon_coords is None:
        raise ValueError(
            "lat_coords and lon_coords must be provided when precipitation "
            "is a numpy array"
        )
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Auto-scale vmax if not provided
    if vmax is None:
        vmax = np.nanmax(data)
        if vmax == 0:
            vmax = 1.0  # Avoid division by zero
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    # Plot precipitation map
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label(colorbar_label, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    return ax


def create_comparison_plot(
    ground_truth: Union[xr.DataArray, np.ndarray],
    experiment1: Union[xr.DataArray, np.ndarray],
    experiment2: Union[xr.DataArray, np.ndarray],
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    exp1_name: str = "Experiment 1 (Baseline)",
    exp2_name: str = "Experiment 2 (With Upstream)",
    title: str = "Precipitation Comparison",
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (18, 5)
) -> plt.Figure:
    """Create side-by-side comparison plots of ground truth and two experiments.
    
    Generates a 1x3 grid showing ground truth, experiment 1 prediction, and
    experiment 2 prediction. All three plots use the same colormap scale for
    fair comparison.
    
    Args:
        ground_truth: Ground truth precipitation [H, W].
        experiment1: Experiment 1 prediction [H, W].
        experiment2: Experiment 2 prediction [H, W].
        lat_coords: Latitude coordinates (required if inputs are numpy arrays).
        lon_coords: Longitude coordinates (required if inputs are numpy arrays).
        exp1_name: Name for experiment 1 (default: 'Experiment 1 (Baseline)').
        exp2_name: Name for experiment 2 (default: 'Experiment 2 (With Upstream)').
        title: Overall title for the figure.
        cmap: Matplotlib colormap name (default: 'Blues').
        vmin: Minimum value for colormap (default: 0.0).
        vmax: Maximum value for colormap (default: auto-scale to max across all plots).
        figsize: Figure size as (width, height) tuple.
    
    Returns:
        Matplotlib figure with comparison plots.
    
    Example:
        >>> fig = create_comparison_plot(
        ...     ground_truth=truth.precipitation,
        ...     experiment1=pred1.precipitation,
        ...     experiment2=pred2.precipitation,
        ...     exp1_name="Baseline (Downstream Only)",
        ...     exp2_name="With Upstream Data",
        ...     title="Precipitation Comparison - 2020-01-15"
        ... )
        >>> fig.savefig('comparison.png', dpi=150, bbox_inches='tight')
        >>> plt.close(fig)
    """
    # Extract coordinates if using xarray
    if isinstance(ground_truth, xr.DataArray):
        if lat_coords is None:
            lat_coords = ground_truth.lat.values
        if lon_coords is None:
            lon_coords = ground_truth.lon.values
    
    # Auto-scale vmax to maximum across all three datasets
    if vmax is None:
        gt_data = ground_truth.values if isinstance(ground_truth, xr.DataArray) else ground_truth
        exp1_data = experiment1.values if isinstance(experiment1, xr.DataArray) else experiment1
        exp2_data = experiment2.values if isinstance(experiment2, xr.DataArray) else experiment2
        
        vmax = max(
            np.nanmax(gt_data),
            np.nanmax(exp1_data),
            np.nanmax(exp2_data)
        )
        if vmax == 0:
            vmax = 1.0
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot ground truth
    plot_precipitation_map(
        precipitation=ground_truth,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        title="Ground Truth",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
        show_colorbar=True
    )
    
    # Plot experiment 1
    plot_precipitation_map(
        precipitation=experiment1,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        title=exp1_name,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=axes[1],
        show_colorbar=True
    )
    
    # Plot experiment 2
    plot_precipitation_map(
        precipitation=experiment2,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        title=exp2_name,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=axes[2],
        show_colorbar=True
    )
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_error_map(
    prediction: Union[xr.DataArray, np.ndarray],
    ground_truth: Union[xr.DataArray, np.ndarray],
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    title: str = "Prediction Error",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    colorbar_label: str = "Error (mm)"
) -> Tuple[plt.Axes, np.ndarray]:
    """Create error/difference map showing prediction - ground_truth.
    
    Generates a map showing the difference between predictions and ground truth.
    Uses a diverging colormap (red for overprediction, blue for underprediction)
    centered at zero.
    
    Args:
        prediction: Predicted precipitation [H, W].
        ground_truth: Ground truth precipitation [H, W].
        lat_coords: Latitude coordinates (required if inputs are numpy arrays).
        lon_coords: Longitude coordinates (required if inputs are numpy arrays).
        title: Title for the plot.
        cmap: Matplotlib colormap name (default: 'RdBu_r' - diverging).
        vmin: Minimum value for colormap (default: symmetric around 0).
        vmax: Maximum value for colormap (default: symmetric around 0).
        ax: Matplotlib axes to plot on. If None, creates new figure.
        show_colorbar: Whether to show colorbar (default: True).
        colorbar_label: Label for colorbar (default: 'Error (mm)').
    
    Returns:
        Tuple of (axes, error_array):
        - axes: Matplotlib axes with the plot
        - error_array: 2D numpy array of errors (prediction - ground_truth)
    
    Example:
        >>> ax, errors = create_error_map(
        ...     prediction=pred.precipitation,
        ...     ground_truth=truth.precipitation,
        ...     title="Prediction Error - Experiment 1"
        ... )
        >>> print(f"Mean error: {np.mean(errors):.2f} mm")
        >>> print(f"RMSE: {np.sqrt(np.mean(errors**2)):.2f} mm")
        >>> plt.savefig('error_map.png')
        >>> plt.close()
    """
    # Extract data and coordinates
    if isinstance(prediction, xr.DataArray):
        pred_data = prediction.values
        if lat_coords is None:
            lat_coords = prediction.lat.values
        if lon_coords is None:
            lon_coords = prediction.lon.values
    else:
        pred_data = prediction
    
    if isinstance(ground_truth, xr.DataArray):
        truth_data = ground_truth.values
    else:
        truth_data = ground_truth
    
    # Validate coordinates
    if lat_coords is None or lon_coords is None:
        raise ValueError(
            "lat_coords and lon_coords must be provided when inputs "
            "are numpy arrays"
        )
    
    # Compute error (prediction - ground_truth)
    error = pred_data - truth_data
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Auto-scale vmin/vmax symmetrically around 0
    if vmin is None or vmax is None:
        max_abs_error = np.nanmax(np.abs(error))
        if max_abs_error == 0:
            max_abs_error = 1.0
        vmin = -max_abs_error
        vmax = max_abs_error
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    # Plot error map
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        error,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label(colorbar_label, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add zero contour line to highlight areas of no error
    ax.contour(
        lon_grid,
        lat_grid,
        error,
        levels=[0],
        colors='black',
        linewidths=1.5,
        linestyles='--',
        alpha=0.5
    )
    
    return ax, error


def save_plot(
    fig: plt.Figure,
    output_dir: Union[str, Path],
    experiment_name: str,
    timestep: Optional[Union[int, str]] = None,
    plot_type: str = "prediction",
    dpi: int = 150,
    format: str = "png"
) -> Path:
    """Save plot with descriptive filename.
    
    Generates a descriptive filename based on experiment name, timestep, and
    plot type, then saves the figure to the specified directory.
    
    Args:
        fig: Matplotlib figure to save.
        output_dir: Directory to save plot in.
        experiment_name: Name of the experiment (e.g., 'baseline', 'with_upstream').
        timestep: Timestep identifier (e.g., 0, '2020-01-15', None for summary plots).
        plot_type: Type of plot ('prediction', 'comparison', 'error', 'summary').
        dpi: Resolution in dots per inch (default: 150).
        format: File format ('png', 'pdf', 'jpg', etc.).
    
    Returns:
        Path to saved file.
    
    Example:
        >>> fig = create_comparison_plot(...)
        >>> filepath = save_plot(
        ...     fig=fig,
        ...     output_dir='outputs/visualizations',
        ...     experiment_name='baseline_vs_upstream',
        ...     timestep='2020-01-15',
        ...     plot_type='comparison'
        ... )
        >>> print(f"Saved to {filepath}")
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    filename_parts = [experiment_name, plot_type]
    
    if timestep is not None:
        # Convert timestep to string and sanitize
        timestep_str = str(timestep).replace(':', '-').replace(' ', '_')
        filename_parts.append(timestep_str)
    
    filename = '_'.join(filename_parts) + f'.{format}'
    filepath = output_dir / filename
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
    logger.info(f"Saved plot to {filepath}")
    
    return filepath


def create_multi_error_comparison(
    predictions_list: List[Union[xr.DataArray, np.ndarray]],
    ground_truth: Union[xr.DataArray, np.ndarray],
    experiment_names: List[str],
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    title: str = "Error Comparison Across Experiments",
    cmap: str = "RdBu_r",
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """Create side-by-side error maps for multiple experiments.
    
    Generates a grid of error maps comparing multiple experiments against
    the same ground truth. Useful for visualizing how different models or
    configurations perform on the same data.
    
    Args:
        predictions_list: List of prediction arrays, one per experiment.
        ground_truth: Ground truth precipitation [H, W].
        experiment_names: List of names for each experiment.
        lat_coords: Latitude coordinates (required if inputs are numpy arrays).
        lon_coords: Longitude coordinates (required if inputs are numpy arrays).
        title: Overall title for the figure.
        cmap: Matplotlib colormap name (default: 'RdBu_r').
        figsize: Figure size as (width, height) tuple. If None, auto-calculated.
    
    Returns:
        Matplotlib figure with error comparison plots.
    
    Raises:
        ValueError: If predictions_list and experiment_names have different lengths.
    
    Example:
        >>> fig = create_multi_error_comparison(
        ...     predictions_list=[pred1.precipitation, pred2.precipitation],
        ...     ground_truth=truth.precipitation,
        ...     experiment_names=['Baseline', 'With Upstream'],
        ...     title='Error Comparison - 2020-01-15'
        ... )
        >>> fig.savefig('error_comparison.png', dpi=150, bbox_inches='tight')
        >>> plt.close(fig)
    """
    if len(predictions_list) != len(experiment_names):
        raise ValueError(
            f"predictions_list ({len(predictions_list)}) and experiment_names "
            f"({len(experiment_names)}) must have the same length"
        )
    
    num_experiments = len(predictions_list)
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (6 * num_experiments, 5)
    
    # Extract coordinates if using xarray
    if isinstance(ground_truth, xr.DataArray):
        if lat_coords is None:
            lat_coords = ground_truth.lat.values
        if lon_coords is None:
            lon_coords = ground_truth.lon.values
    
    # Calculate global vmin/vmax for consistent color scale
    all_errors = []
    for pred in predictions_list:
        pred_data = pred.values if isinstance(pred, xr.DataArray) else pred
        truth_data = ground_truth.values if isinstance(ground_truth, xr.DataArray) else ground_truth
        error = pred_data - truth_data
        all_errors.append(error)
    
    max_abs_error = max(np.nanmax(np.abs(err)) for err in all_errors)
    if max_abs_error == 0:
        max_abs_error = 1.0
    vmin = -max_abs_error
    vmax = max_abs_error
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_experiments, figsize=figsize)
    
    # Handle single experiment case (axes is not a list)
    if num_experiments == 1:
        axes = [axes]
    
    # Plot each error map
    for i, (pred, exp_name) in enumerate(zip(predictions_list, experiment_names)):
        create_error_map(
            prediction=pred,
            ground_truth=ground_truth,
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            title=f"{exp_name} Error",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=axes[i],
            show_colorbar=True
        )
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
