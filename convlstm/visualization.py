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



# ============================================================================
# Multi-Variable Visualization Functions
# ============================================================================

# Atmospheric variable metadata
ATMOSPHERIC_VARIABLES = {
    'DPT': {
        'name': 'Dew Point Temperature',
        'unit': 'K',
        'cmap': 'RdYlBu_r',
        'channel_start': 0,
        'channel_end': 11
    },
    'GPH': {
        'name': 'Geopotential Height',
        'unit': 'm',
        'cmap': 'viridis',
        'channel_start': 11,
        'channel_end': 22
    },
    'TEM': {
        'name': 'Temperature',
        'unit': 'K',
        'cmap': 'RdYlBu_r',
        'channel_start': 22,
        'channel_end': 33
    },
    'U': {
        'name': 'Eastward Wind',
        'unit': 'm/s',
        'cmap': 'RdBu_r',
        'channel_start': 33,
        'channel_end': 44
    },
    'V': {
        'name': 'Northward Wind',
        'unit': 'm/s',
        'cmap': 'RdBu_r',
        'channel_start': 44,
        'channel_end': 55
    }
}

# Pressure levels in hPa (11 levels from 1000 to 50 hPa)
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 50]

# Representative pressure levels for visualization
REPRESENTATIVE_LEVELS = [850, 500, 200]  # Lower, middle, upper atmosphere


def get_variable_channel_index(variable: str, pressure_level: int) -> int:
    """Get the channel index for a specific variable and pressure level.
    
    Args:
        variable: Variable name ('DPT', 'GPH', 'TEM', 'U', 'V')
        pressure_level: Pressure level in hPa (must be in PRESSURE_LEVELS)
    
    Returns:
        Channel index (0-54 for atmospheric variables, 55 for precipitation)
    
    Raises:
        ValueError: If variable or pressure level is invalid
    
    Example:
        >>> idx = get_variable_channel_index('TEM', 850)
        >>> print(f"Temperature at 850 hPa is channel {idx}")
    """
    if variable == 'precipitation':
        return 55
    
    if variable not in ATMOSPHERIC_VARIABLES:
        raise ValueError(
            f"Invalid variable '{variable}'. Must be one of: "
            f"{list(ATMOSPHERIC_VARIABLES.keys())} or 'precipitation'"
        )
    
    if pressure_level not in PRESSURE_LEVELS:
        raise ValueError(
            f"Invalid pressure level {pressure_level}. Must be one of: {PRESSURE_LEVELS}"
        )
    
    var_info = ATMOSPHERIC_VARIABLES[variable]
    level_idx = PRESSURE_LEVELS.index(pressure_level)
    channel_idx = var_info['channel_start'] + level_idx
    
    return channel_idx


def plot_atmospheric_variable(
    data: Union[xr.DataArray, np.ndarray],
    variable: str,
    pressure_level: int,
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Axes:
    """Plot a single atmospheric variable at a specific pressure level.
    
    Args:
        data: 2D array [H, W] or multi-channel array [C, H, W] or xarray DataArray
        variable: Variable name ('DPT', 'GPH', 'TEM', 'U', 'V', 'precipitation')
        pressure_level: Pressure level in hPa (ignored for precipitation)
        lat_coords: Latitude coordinates (required if data is numpy array)
        lon_coords: Longitude coordinates (required if data is numpy array)
        title: Custom title (auto-generated if None)
        ax: Matplotlib axes to plot on (creates new if None)
        show_colorbar: Whether to show colorbar
        vmin: Minimum value for colormap (auto-scaled if None)
        vmax: Maximum value for colormap (auto-scaled if None)
    
    Returns:
        Matplotlib axes with the plot
    
    Example:
        >>> # Plot temperature at 850 hPa
        >>> ax = plot_atmospheric_variable(
        ...     data=predictions,  # [56, H, W]
        ...     variable='TEM',
        ...     pressure_level=850,
        ...     lat_coords=lats,
        ...     lon_coords=lons
        ... )
        >>> plt.savefig('temperature_850hPa.png')
        >>> plt.close()
    """
    # Extract data and coordinates
    if isinstance(data, xr.DataArray):
        data_array = data.values
        if lat_coords is None and 'lat' in data.dims:
            lat_coords = data.lat.values
        if lon_coords is None and 'lon' in data.dims:
            lon_coords = data.lon.values
    else:
        data_array = data
    
    # Validate coordinates
    if lat_coords is None or lon_coords is None:
        raise ValueError(
            "lat_coords and lon_coords must be provided when data is a numpy array"
        )
    
    # Extract the specific variable/level
    if variable == 'precipitation':
        if data_array.ndim == 3:
            plot_data = data_array[55, :, :]
        else:
            plot_data = data_array
        var_name = 'Precipitation'
        var_unit = 'mm'
        cmap = 'Blues'
    else:
        channel_idx = get_variable_channel_index(variable, pressure_level)
        if data_array.ndim == 3:
            plot_data = data_array[channel_idx, :, :]
        else:
            plot_data = data_array
        
        var_info = ATMOSPHERIC_VARIABLES[variable]
        var_name = var_info['name']
        var_unit = var_info['unit']
        cmap = var_info['cmap']
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Auto-scale vmin/vmax if not provided
    if vmin is None:
        vmin = np.nanmin(plot_data)
    if vmax is None:
        vmax = np.nanmax(plot_data)
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    # Plot the variable
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        plot_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label(f'{var_name} ({var_unit})', fontsize=12)
    
    # Set title
    if title is None:
        if variable == 'precipitation':
            title = f'{var_name}'
        else:
            title = f'{var_name} at {pressure_level} hPa'
    
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    return ax


def create_multi_variable_comparison(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    variable: str,
    pressure_levels: Optional[List[int]] = None,
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """Create side-by-side comparison of prediction vs ground truth for a variable.
    
    Shows the variable at multiple pressure levels (or just precipitation) with
    prediction, ground truth, and error for each level.
    
    Args:
        prediction: Prediction array [56, H, W]
        ground_truth: Ground truth array [56, H, W]
        variable: Variable name ('DPT', 'GPH', 'TEM', 'U', 'V', 'precipitation')
        pressure_levels: List of pressure levels to plot (uses REPRESENTATIVE_LEVELS if None)
        lat_coords: Latitude coordinates
        lon_coords: Longitude coordinates
        title: Overall title (auto-generated if None)
        figsize: Figure size (auto-calculated if None)
    
    Returns:
        Matplotlib figure with comparison plots
    
    Example:
        >>> fig = create_multi_variable_comparison(
        ...     prediction=pred_array,  # [56, H, W]
        ...     ground_truth=truth_array,  # [56, H, W]
        ...     variable='TEM',
        ...     lat_coords=lats,
        ...     lon_coords=lons
        ... )
        >>> fig.savefig('temperature_comparison.png', dpi=150, bbox_inches='tight')
        >>> plt.close(fig)
    """
    if variable == 'precipitation':
        pressure_levels = [None]
    elif pressure_levels is None:
        pressure_levels = REPRESENTATIVE_LEVELS
    
    num_levels = len(pressure_levels)
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (18, 5 * num_levels)
    
    # Create figure with 3 columns (prediction, truth, error) and num_levels rows
    fig, axes = plt.subplots(num_levels, 3, figsize=figsize)
    
    # Handle single level case
    if num_levels == 1:
        axes = axes.reshape(1, -1)
    
    # Get variable info
    if variable == 'precipitation':
        var_name = 'Precipitation'
        cmap = 'Blues'
        error_cmap = 'RdBu_r'
    else:
        var_info = ATMOSPHERIC_VARIABLES[variable]
        var_name = var_info['name']
        cmap = var_info['cmap']
        error_cmap = 'RdBu_r'
    
    # Plot each pressure level
    for i, level in enumerate(pressure_levels):
        # Extract data for this level
        if variable == 'precipitation':
            pred_data = prediction[55, :, :]
            truth_data = ground_truth[55, :, :]
            level_str = ''
        else:
            channel_idx = get_variable_channel_index(variable, level)
            pred_data = prediction[channel_idx, :, :]
            truth_data = ground_truth[channel_idx, :, :]
            level_str = f' at {level} hPa'
        
        # Calculate error
        error_data = pred_data - truth_data
        
        # Determine color scale (same for prediction and truth)
        vmin = min(np.nanmin(pred_data), np.nanmin(truth_data))
        vmax = max(np.nanmax(pred_data), np.nanmax(truth_data))
        
        # Error scale (symmetric around 0)
        max_abs_error = np.nanmax(np.abs(error_data))
        if max_abs_error == 0:
            max_abs_error = 1.0
        error_vmin = -max_abs_error
        error_vmax = max_abs_error
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        
        # Plot prediction
        im1 = axes[i, 0].pcolormesh(
            lon_grid, lat_grid, pred_data,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )
        axes[i, 0].set_title(f'Prediction{level_str}', fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('Longitude (°E)')
        axes[i, 0].set_ylabel('Latitude (°N)')
        axes[i, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.colorbar(im1, ax=axes[i, 0], orientation='vertical', pad=0.02)
        
        # Plot ground truth
        im2 = axes[i, 1].pcolormesh(
            lon_grid, lat_grid, truth_data,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )
        axes[i, 1].set_title(f'Ground Truth{level_str}', fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('Longitude (°E)')
        axes[i, 1].set_ylabel('Latitude (°N)')
        axes[i, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.colorbar(im2, ax=axes[i, 1], orientation='vertical', pad=0.02)
        
        # Plot error
        im3 = axes[i, 2].pcolormesh(
            lon_grid, lat_grid, error_data,
            cmap=error_cmap, vmin=error_vmin, vmax=error_vmax, shading='auto'
        )
        axes[i, 2].set_title(f'Error{level_str}', fontsize=12, fontweight='bold')
        axes[i, 2].set_xlabel('Longitude (°E)')
        axes[i, 2].set_ylabel('Latitude (°N)')
        axes[i, 2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[i, 2].contour(
            lon_grid, lat_grid, error_data,
            levels=[0], colors='black', linewidths=1.5,
            linestyles='--', alpha=0.5
        )
        plt.colorbar(im3, ax=axes[i, 2], orientation='vertical', pad=0.02)
    
    # Set overall title
    if title is None:
        title = f'{var_name} Comparison'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    return fig


def create_rolling_forecast_evolution(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    variable: str,
    pressure_level: Optional[int] = None,
    lat_coords: Optional[np.ndarray] = None,
    lon_coords: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """Create evolution plot showing rolling forecast across multiple timesteps.
    
    Shows how predictions evolve over time in a rolling forecast, with one row
    per timestep showing prediction, ground truth, and error.
    
    Args:
        predictions: Prediction array [num_steps, 56, H, W]
        ground_truth: Ground truth array [num_steps, 56, H, W]
        variable: Variable name ('DPT', 'GPH', 'TEM', 'U', 'V', 'precipitation')
        pressure_level: Pressure level in hPa (required for atmospheric variables)
        lat_coords: Latitude coordinates
        lon_coords: Longitude coordinates
        title: Overall title (auto-generated if None)
        figsize: Figure size (auto-calculated if None)
    
    Returns:
        Matplotlib figure with rolling forecast evolution
    
    Example:
        >>> fig = create_rolling_forecast_evolution(
        ...     predictions=rolling_preds,  # [6, 56, H, W]
        ...     ground_truth=rolling_truth,  # [6, 56, H, W]
        ...     variable='precipitation',
        ...     lat_coords=lats,
        ...     lon_coords=lons
        ... )
        >>> fig.savefig('rolling_forecast_evolution.png', dpi=150, bbox_inches='tight')
        >>> plt.close(fig)
    """
    num_steps = predictions.shape[0]
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (18, 5 * num_steps)
    
    # Create figure with 3 columns and num_steps rows
    fig, axes = plt.subplots(num_steps, 3, figsize=figsize)
    
    # Handle single step case
    if num_steps == 1:
        axes = axes.reshape(1, -1)
    
    # Get variable info
    if variable == 'precipitation':
        var_name = 'Precipitation'
        cmap = 'Blues'
        channel_idx = 55
    else:
        if pressure_level is None:
            raise ValueError(
                f"pressure_level is required for atmospheric variable '{variable}'"
            )
        var_info = ATMOSPHERIC_VARIABLES[variable]
        var_name = var_info['name']
        cmap = var_info['cmap']
        channel_idx = get_variable_channel_index(variable, pressure_level)
    
    error_cmap = 'RdBu_r'
    
    # Calculate global color scale across all timesteps
    all_pred_data = predictions[:, channel_idx, :, :]
    all_truth_data = ground_truth[:, channel_idx, :, :]
    vmin = min(np.nanmin(all_pred_data), np.nanmin(all_truth_data))
    vmax = max(np.nanmax(all_pred_data), np.nanmax(all_truth_data))
    
    all_errors = all_pred_data - all_truth_data
    max_abs_error = np.nanmax(np.abs(all_errors))
    if max_abs_error == 0:
        max_abs_error = 1.0
    error_vmin = -max_abs_error
    error_vmax = max_abs_error
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
    
    # Plot each timestep
    for step in range(num_steps):
        pred_data = predictions[step, channel_idx, :, :]
        truth_data = ground_truth[step, channel_idx, :, :]
        error_data = pred_data - truth_data
        
        # Plot prediction
        im1 = axes[step, 0].pcolormesh(
            lon_grid, lat_grid, pred_data,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )
        axes[step, 0].set_title(f'Prediction (t+{step+1})', fontsize=12, fontweight='bold')
        axes[step, 0].set_xlabel('Longitude (°E)')
        axes[step, 0].set_ylabel('Latitude (°N)')
        axes[step, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.colorbar(im1, ax=axes[step, 0], orientation='vertical', pad=0.02)
        
        # Plot ground truth
        im2 = axes[step, 1].pcolormesh(
            lon_grid, lat_grid, truth_data,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )
        axes[step, 1].set_title(f'Ground Truth (t+{step+1})', fontsize=12, fontweight='bold')
        axes[step, 1].set_xlabel('Longitude (°E)')
        axes[step, 1].set_ylabel('Latitude (°N)')
        axes[step, 1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.colorbar(im2, ax=axes[step, 1], orientation='vertical', pad=0.02)
        
        # Plot error
        im3 = axes[step, 2].pcolormesh(
            lon_grid, lat_grid, error_data,
            cmap=error_cmap, vmin=error_vmin, vmax=error_vmax, shading='auto'
        )
        axes[step, 2].set_title(f'Error (t+{step+1})', fontsize=12, fontweight='bold')
        axes[step, 2].set_xlabel('Longitude (°E)')
        axes[step, 2].set_ylabel('Latitude (°N)')
        axes[step, 2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[step, 2].contour(
            lon_grid, lat_grid, error_data,
            levels=[0], colors='black', linewidths=1.5,
            linestyles='--', alpha=0.5
        )
        plt.colorbar(im3, ax=axes[step, 2], orientation='vertical', pad=0.02)
    
    # Set overall title
    if title is None:
        if variable == 'precipitation':
            title = f'{var_name} - Rolling Forecast Evolution'
        else:
            title = f'{var_name} at {pressure_level} hPa - Rolling Forecast Evolution'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    return fig


def save_multi_variable_plot(
    fig: plt.Figure,
    output_dir: Union[str, Path],
    variable: str,
    timestep: Optional[Union[int, str]] = None,
    pressure_level: Optional[int] = None,
    plot_type: str = "comparison",
    experiment_name: Optional[str] = None,
    dpi: int = 150,
    format: str = "png"
) -> Path:
    """Save multi-variable plot with organized filename structure.
    
    Organizes output files by variable type and timestep for easy navigation.
    Creates subdirectories for each variable type.
    
    Args:
        fig: Matplotlib figure to save
        output_dir: Base output directory
        variable: Variable name ('DPT', 'GPH', 'TEM', 'U', 'V', 'precipitation')
        timestep: Timestep identifier (e.g., 0, 't+1', '2020-01-15')
        pressure_level: Pressure level in hPa (included in filename if provided)
        plot_type: Type of plot ('comparison', 'evolution', 'error', 'prediction')
        experiment_name: Optional experiment name to include in filename
        dpi: Resolution in dots per inch
        format: File format ('png', 'pdf', 'jpg', etc.)
    
    Returns:
        Path to saved file
    
    Example:
        >>> fig = create_multi_variable_comparison(...)
        >>> filepath = save_multi_variable_plot(
        ...     fig=fig,
        ...     output_dir='outputs/visualizations',
        ...     variable='TEM',
        ...     timestep='t+1',
        ...     pressure_level=850,
        ...     plot_type='comparison',
        ...     experiment_name='baseline'
        ... )
        >>> print(f"Saved to {filepath}")
        # Output: outputs/visualizations/TEM/baseline_TEM_850hPa_comparison_t+1.png
    """
    # Create base output directory
    output_dir = Path(output_dir)
    
    # Create subdirectory for this variable
    var_dir = output_dir / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename parts
    filename_parts = []
    
    if experiment_name:
        filename_parts.append(experiment_name)
    
    filename_parts.append(variable)
    
    if pressure_level is not None:
        filename_parts.append(f'{pressure_level}hPa')
    
    filename_parts.append(plot_type)
    
    if timestep is not None:
        timestep_str = str(timestep).replace(':', '-').replace(' ', '_')
        filename_parts.append(timestep_str)
    
    filename = '_'.join(filename_parts) + f'.{format}'
    filepath = var_dir / filename
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
    logger.info(f"Saved multi-variable plot to {filepath}")
    
    return filepath


def visualize_all_variables(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    output_dir: Union[str, Path],
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    timestep: Optional[Union[int, str]] = None,
    experiment_name: Optional[str] = None,
    variables: Optional[List[str]] = None,
    pressure_levels: Optional[List[int]] = None,
    dpi: int = 150
) -> List[Path]:
    """Visualize all atmospheric variables and precipitation.
    
    Creates comparison plots for all specified variables and saves them in
    organized subdirectories. This is a convenience function for generating
    a complete set of visualizations.
    
    Args:
        prediction: Prediction array [56, H, W]
        ground_truth: Ground truth array [56, H, W]
        output_dir: Base output directory
        lat_coords: Latitude coordinates
        lon_coords: Longitude coordinates
        timestep: Timestep identifier
        experiment_name: Experiment name for filenames
        variables: List of variables to plot (defaults to all + precipitation)
        pressure_levels: Pressure levels to plot (defaults to REPRESENTATIVE_LEVELS)
        dpi: Resolution for saved figures
    
    Returns:
        List of paths to saved figures
    
    Example:
        >>> saved_files = visualize_all_variables(
        ...     prediction=pred_array,
        ...     ground_truth=truth_array,
        ...     output_dir='outputs/visualizations',
        ...     lat_coords=lats,
        ...     lon_coords=lons,
        ...     timestep='t+1',
        ...     experiment_name='baseline'
        ... )
        >>> print(f"Created {len(saved_files)} visualization files")
    """
    if variables is None:
        variables = list(ATMOSPHERIC_VARIABLES.keys()) + ['precipitation']
    
    if pressure_levels is None:
        pressure_levels = REPRESENTATIVE_LEVELS
    
    saved_paths = []
    
    for variable in variables:
        logger.info(f"Creating visualization for {variable}...")
        
        try:
            if variable == 'precipitation':
                # Create precipitation comparison
                fig = create_multi_variable_comparison(
                    prediction=prediction,
                    ground_truth=ground_truth,
                    variable='precipitation',
                    lat_coords=lat_coords,
                    lon_coords=lon_coords
                )
                
                filepath = save_multi_variable_plot(
                    fig=fig,
                    output_dir=output_dir,
                    variable='precipitation',
                    timestep=timestep,
                    plot_type='comparison',
                    experiment_name=experiment_name,
                    dpi=dpi
                )
                saved_paths.append(filepath)
                plt.close(fig)
            else:
                # Create atmospheric variable comparison
                fig = create_multi_variable_comparison(
                    prediction=prediction,
                    ground_truth=ground_truth,
                    variable=variable,
                    pressure_levels=pressure_levels,
                    lat_coords=lat_coords,
                    lon_coords=lon_coords
                )
                
                filepath = save_multi_variable_plot(
                    fig=fig,
                    output_dir=output_dir,
                    variable=variable,
                    timestep=timestep,
                    plot_type='comparison',
                    experiment_name=experiment_name,
                    dpi=dpi
                )
                saved_paths.append(filepath)
                plt.close(fig)
        
        except Exception as e:
            logger.error(f"Failed to create visualization for {variable}: {e}")
            continue
    
    logger.info(f"Created {len(saved_paths)} visualization files in {output_dir}")
    
    return saved_paths
