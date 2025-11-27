"""Data processing and dataset classes for ConvLSTM weather prediction.

This module implements data processing utilities and PyTorch Dataset classes
for loading and preparing atmospheric data for ConvLSTM training.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


# Import constants directly to avoid JAX dependency
HPA_VARIABLES = ["DPT", "GPH", "TEM", "U", "V"]
PRESSURE_LEVELS = np.array([100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000])

# Variable name mapping for different data formats
# Maps from common names to expected names
VARIABLE_NAME_MAPPING = {
    'dew_point': 'DPT',
    'geopotential_height': 'GPH',
    'temperature': 'TEM',
    'u_wind': 'U',
    'v_wind': 'V',
    # Also support the original names (identity mapping)
    'DPT': 'DPT',
    'GPH': 'GPH',
    'TEM': 'TEM',
    'U': 'U',
    'V': 'V'
}


@dataclass
class RegionConfig:
    """Configuration for regional boundaries.
    
    This is a simplified version of graphcast_regional.config.RegionConfig
    that avoids JAX dependencies.
    
    Attributes:
        downstream_lat_min: Minimum latitude for downstream region (degrees).
        downstream_lat_max: Maximum latitude for downstream region (degrees).
        downstream_lon_min: Minimum longitude for downstream region (degrees).
        downstream_lon_max: Maximum longitude for downstream region (degrees).
        upstream_lat_min: Minimum latitude for upstream region (degrees).
        upstream_lat_max: Maximum latitude for upstream region (degrees).
        upstream_lon_min: Minimum longitude for upstream region (degrees).
        upstream_lon_max: Maximum longitude for upstream region (degrees).
    """
    # Downstream region boundaries (target prediction area)
    downstream_lat_min: float = 25.0
    downstream_lat_max: float = 40.0
    downstream_lon_min: float = 110.0
    downstream_lon_max: float = 125.0
    
    # Upstream region boundaries (influencing area)
    upstream_lat_min: float = 25.0
    upstream_lat_max: float = 50.0
    upstream_lon_min: float = 70.0
    upstream_lon_max: float = 110.0


logger = logging.getLogger(__name__)


def create_train_val_test_split(
    data: xr.Dataset,
    train_end_year: Optional[int] = None,
    val_end_year: Optional[int] = None,
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
    test_start_date: Optional[str] = None,
    trainval_end_date: Optional[str] = None,
    random_trainval_split: bool = True,
    random_seed: int = 42,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Split data into train/val/test sets with flexible splitting strategies.
    
    This function supports two splitting strategies:
    1. Random split for train/val (when random_trainval_split=True):
       - Test set is split by time (temporal cutoff)
       - Train/val are randomly sampled from remaining data
    2. Sequential split (when random_trainval_split=False):
       - All splits maintain temporal ordering (original behavior)
    
    Args:
        data: Full dataset with time dimension.
        train_end_year: Last year (inclusive) for training data. If None, uses ratios or dates.
        val_end_year: Last year (inclusive) for validation data. If None, uses ratios or dates.
        train_ratio: Fraction of data for training (if years not specified).
        val_ratio: Fraction of data for validation (if years not specified).
        test_start_date: Date string (YYYY-MM-DD) when test set starts. If specified,
                        data before this date is split into train/val.
        trainval_end_date: Date string (YYYY-MM-DD) when train/val data ends. If specified,
                          only data before this date is used for training and validation,
                          and data from this date onwards is used for testing.
        random_trainval_split: If True, randomly split train/val while keeping test temporal.
                              If False, use sequential temporal splitting (default: True).
        random_seed: Random seed for reproducible train/val splitting (default: 42).
        
    Returns:
        Tuple of (train_data, val_data, test_data).
        
    Raises:
        ValueError: If no training data is found or if temporal ordering is violated.
        
    Examples:
        >>> # Random train/val split with temporal test split
        >>> train, val, test = create_train_val_test_split(
        ...     data, test_start_date='2020-01-01', random_trainval_split=True
        ... )
        
        >>> # Sequential temporal splitting (original behavior)
        >>> train, val, test = create_train_val_test_split(
        ...     data, test_start_date='2020-01-01', random_trainval_split=False
        ... )
    """
    logger.info("Splitting data into train/val/test sets...")
    
    # Extract time coordinates
    times = data.time.values
    
    # Log the actual time range in the data
    logger.info(f"Data time range: {times.min()} to {times.max()}")
    logger.info(f"Total timesteps: {len(times)}")
    
    # Validate temporal ordering in input data
    if not np.all(times[:-1] <= times[1:]):
        raise ValueError(
            "Input data does not have monotonically increasing time coordinates. "
            "Temporal ordering must be preserved for time series prediction."
        )
    
    # Step 1: Split test set by time
    if trainval_end_date is not None:
        logger.info(f"Using trainval_end_date: train/val data before {trainval_end_date}, test data from {trainval_end_date} onwards")
        test_cutoff = np.datetime64(trainval_end_date)
    elif test_start_date is not None:
        logger.info(f"Using test_start_date: test starts at {test_start_date}")
        test_cutoff = np.datetime64(test_start_date)
    elif train_end_year is not None and val_end_year is not None:
        logger.info(f"Using year-based split: test starts after {val_end_year}")
        test_cutoff = np.datetime64(f'{val_end_year + 1}-01-01')
    else:
        # No explicit test cutoff, use ratio-based splitting
        logger.info(f"Using ratio-based split: train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio}")
        n_total = len(times)
        n_trainval = int(n_total * (train_ratio + val_ratio))
        test_cutoff = times[n_trainval]
        logger.info(f"Test cutoff determined by ratio: {test_cutoff}")
    
    # Split into train+val and test by time
    trainval_mask = times < test_cutoff
    test_mask = times >= test_cutoff
    
    trainval_data = data.isel(time=trainval_mask)
    test_data = data.isel(time=test_mask)
    
    logger.info(f"Test cutoff: {test_cutoff}")
    logger.info(f"Train+Val timesteps: {len(trainval_data.time)}")
    logger.info(f"Test timesteps: {len(test_data.time)}")
    
    # Validate that we have training data
    if len(trainval_data.time) == 0:
        raise ValueError(
            f"No training data found. Data time range is {times.min()} to {times.max()}. "
            f"Test cutoff is {test_cutoff}. Please adjust split parameters."
        )
    
    # Step 2: Split train and val
    n_trainval = len(trainval_data.time)
    n_train = int(n_trainval * train_ratio / (train_ratio + val_ratio))
    n_val = n_trainval - n_train
    
    if random_trainval_split:
        # Random split for train/val
        logger.info(f"Using RANDOM train/val split with seed={random_seed}")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Generate random indices
        indices = np.arange(n_trainval)
        np.random.shuffle(indices)
        
        train_indices = np.sort(indices[:n_train])  # Sort to maintain some temporal locality
        val_indices = np.sort(indices[n_train:])
        
        train_data = trainval_data.isel(time=train_indices)
        val_data = trainval_data.isel(time=val_indices)
        
        logger.info(f"Random split: {n_train} train, {n_val} val samples")
        
    else:
        # Sequential temporal split (original behavior)
        logger.info(f"Using SEQUENTIAL train/val split")
        
        train_data = trainval_data.isel(time=slice(0, n_train))
        val_data = trainval_data.isel(time=slice(n_train, None))
        
        logger.info(f"Sequential split: {n_train} train, {n_val} val samples")
    
    logger.info(f"Train timesteps: {len(train_data.time)}")
    logger.info(f"Val timesteps: {len(val_data.time)}")
    logger.info(f"Test timesteps: {len(test_data.time)}")
    
    # Log time ranges for each split
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        if len(split_data.time) > 0:
            split_times = split_data.time.values
            logger.info(
                f"{split_name.capitalize()} split time range: "
                f"{split_times.min()} to {split_times.max()}"
            )
    
    # Validate test set temporal ordering (must be sequential)
    if len(test_data.time) > 1:
        test_times = test_data.time.values
        if not np.all(test_times[:-1] <= test_times[1:]):
            raise ValueError(
                "Test set temporal ordering violated. "
                "Test data must maintain temporal sequence."
            )
    
    # Validate no overlap between trainval and test
    if len(trainval_data.time) > 0 and len(test_data.time) > 0:
        if trainval_data.time.values.max() >= test_data.time.values.min():
            raise ValueError(
                "Train/Val and test splits overlap in time. "
                "This violates temporal separation for time series prediction."
            )
    
    logger.info("Data splitting completed successfully")
    
    return train_data, val_data, test_data


class ConvLSTMNormalizer:
    """Handles feature normalization for ConvLSTM training and inference.
    
    This normalizer is adapted for ConvLSTM's channel-stacked format and implements
    special handling for precipitation using log1p transformation. It works with
    xarray Datasets and provides normalization/denormalization for both training
    and inference.
    
    Normalization strategy:
    - HPA variables (DPT, GPH, TEM, U, V): Z-score normalization (mean=0, std=1)
    - Precipitation: log1p transformation followed by Z-score normalization
    
    Attributes:
        mean: Mean values for each variable (xarray Dataset).
        std: Standard deviation for each variable (xarray Dataset).
        _is_fitted: Whether the normalizer has been fitted to training data.
    """
    
    def __init__(self):
        """Initialize ConvLSTMNormalizer."""
        self.mean: Optional[xr.Dataset] = None
        self.std: Optional[xr.Dataset] = None
        self._is_fitted = False
    
    def _rename_variables(self, data: xr.Dataset) -> xr.Dataset:
        """Rename variables to match expected names.
        
        Args:
            data: Dataset with potentially different variable names.
            
        Returns:
            Dataset with standardized variable names.
        """
        rename_dict = {}
        for var_name in data.data_vars:
            if var_name in VARIABLE_NAME_MAPPING:
                expected_name = VARIABLE_NAME_MAPPING[var_name]
                if expected_name != var_name:
                    rename_dict[var_name] = expected_name
        
        if rename_dict:
            logger.info(f"Renaming variables: {rename_dict}")
            return data.rename(rename_dict)
        return data
    
    def fit(self, train_data: xr.Dataset, use_parallel: bool = True, n_workers: Optional[int] = None) -> None:
        """Compute normalization statistics from training data with optional parallelization.
        
        Computes mean and standard deviation for HPA variables and log-transformed
        precipitation. Statistics are computed across time, lat, and lon dimensions,
        preserving the level dimension for HPA variables.
        
        Performance tips:
        - For small datasets (<1000 timesteps): use_parallel=False is faster
        - For large datasets (>1000 timesteps): use_parallel=True provides speedup
        - n_workers=None uses all available CPU cores
        
        Args:
            train_data: Training dataset with dimensions (time, level, lat, lon)
                       for HPA variables and (time, lat, lon) for precipitation.
            use_parallel: If True, use parallel computation via NumPy threading (default: True).
                         Automatically disabled for small datasets.
            n_workers: Number of parallel workers. If None, uses all available cores.
                       Only effective when use_parallel=True.
                       
        Raises:
            ValueError: If required variables are missing from train_data.
        """
        import time
        import os
        start_time = time.time()
        
        logger.info("Computing normalization statistics for ConvLSTM format...")
        
        # Rename variables if needed to match expected names
        data_renamed = self._rename_variables(train_data)
        
        # Validate required variables
        for var in HPA_VARIABLES:
            if var not in data_renamed:
                raise ValueError(f"Missing required HPA variable: {var}")
        
        if "precipitation" not in data_renamed:
            raise ValueError("Missing required variable: precipitation")
        
        # Auto-disable parallelization for small datasets (overhead not worth it)
        n_timesteps = len(data_renamed.time)
        if use_parallel and n_timesteps < 1000:
            logger.info(f"Dataset has only {n_timesteps} timesteps, disabling parallelization (overhead > benefit)")
            use_parallel = False
        
        # Set number of threads for NumPy operations
        original_num_threads = {}
        if use_parallel:
            if n_workers is None:
                n_workers = os.cpu_count()
            
            # Configure NumPy/OpenBLAS/MKL threading
            thread_env_vars = [
                'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 
                'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'
            ]
            
            for var in thread_env_vars:
                original_num_threads[var] = os.environ.get(var)
                os.environ[var] = str(n_workers)
            
            logger.info(f"Using parallel computation with {n_workers} threads")
        else:
            logger.info("Using sequential computation")
        
        # Load data into memory if it's lazy-loaded (from NetCDF)
        # This prevents repeated disk I/O during statistics computation
        if hasattr(data_renamed, 'chunks') and data_renamed.chunks:
            logger.info("Loading data into memory (this may take a moment)...")
            data_renamed = data_renamed.load()
            logger.info("✓ Data loaded into memory")
        
        try:
            # Compute statistics for HPA variables directly (no transformation needed)
            logger.info("Computing statistics for HPA variables...")
            mean_dict = {}
            std_dict = {}
            
            for var in HPA_VARIABLES:
                logger.info(f"  Processing {var}...")
                mean_dict[var] = data_renamed[var].mean(dim=["time", "lat", "lon"])
                std_dict[var] = data_renamed[var].std(dim=["time", "lat", "lon"])
            
            # For precipitation: apply log1p transformation before computing statistics
            # log1p(x) = log(1 + x) handles zero values gracefully
            logger.info("  Processing precipitation (with log1p transform)...")
            precip_log = np.log1p(data_renamed["precipitation"])
            mean_dict["precipitation"] = precip_log.mean(dim=["time", "lat", "lon"])
            std_dict["precipitation"] = precip_log.std(dim=["time", "lat", "lon"])
            
            # Combine into xarray Datasets
            self.mean = xr.Dataset(mean_dict)
            self.std = xr.Dataset(std_dict)
            
        finally:
            # Restore original thread settings
            if use_parallel:
                for var, original_value in original_num_threads.items():
                    if original_value is None:
                        os.environ.pop(var, None)
                    else:
                        os.environ[var] = original_value
        
        # Avoid division by zero - replace zero std with 1.0
        for var_name in self.std.data_vars:
            std_values = self.std[var_name].values
            std_values = np.where(std_values == 0, 1.0, std_values)
            self.std[var_name].values = std_values
            
            # Log warning if any constant features detected
            if np.any(std_values == 1.0):
                logger.warning(
                    f"Variable {var_name} has zero standard deviation in some locations. "
                    f"Replaced with 1.0 to avoid division by zero."
                )
        
        self._is_fitted = True
        
        elapsed_time = time.time() - start_time
        logger.info(f"Normalization statistics computed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Variables: {list(self.mean.data_vars)}")
        logger.info(f"Precipitation statistics (log-transformed): "
                   f"mean={float(self.mean['precipitation'].values):.4f}, "
                   f"std={float(self.std['precipitation'].values):.4f}")
    
    def normalize(self, data: xr.Dataset) -> xr.Dataset:
        """Apply normalization to data.
        
        Applies log1p transformation to precipitation followed by Z-score
        normalization for all variables. This ensures the data is in the
        same scale as the training data.
        
        Args:
            data: Dataset to normalize with same structure as training data.
            
        Returns:
            Normalized dataset with same structure as input.
            
        Raises:
            RuntimeError: If normalizer has not been fitted.
            ValueError: If data is missing required variables.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ConvLSTMNormalizer must be fitted before normalizing data. "
                "Call fit() with training data first."
            )
        
        # Rename variables if needed to match expected names
        data_renamed = self._rename_variables(data)
        
        # Validate required variables
        for var in HPA_VARIABLES:
            if var not in data_renamed:
                raise ValueError(f"Missing required HPA variable: {var}")
        
        if "precipitation" not in data_renamed:
            raise ValueError("Missing required variable: precipitation")
        
        # Create a copy to avoid modifying original data
        normalized = data_renamed.copy()
        
        # Apply log1p transformation to precipitation
        normalized["precipitation"] = np.log1p(normalized["precipitation"])
        
        # Apply Z-score normalization: (x - mean) / std
        normalized = (normalized - self.mean) / self.std
        
        return normalized
    
    def denormalize(self, data: xr.Dataset) -> xr.Dataset:
        """Reverse normalization to get original scale.
        
        Reverses Z-score normalization and applies expm1 (inverse of log1p)
        to precipitation to recover original values.
        
        Args:
            data: Normalized dataset.
            
        Returns:
            Denormalized dataset in original scale.
            
        Raises:
            RuntimeError: If normalizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ConvLSTMNormalizer must be fitted before denormalizing data. "
                "Call fit() with training data first."
            )
        
        # Reverse Z-score normalization: x_original = x_normalized * std + mean
        denormalized = data * self.std + self.mean
        
        # Reverse log1p transformation for precipitation: expm1(x) = exp(x) - 1
        denormalized["precipitation"] = np.expm1(denormalized["precipitation"])
        
        # Ensure precipitation is non-negative (clip small negative values from numerical errors)
        denormalized["precipitation"] = denormalized["precipitation"].clip(min=0.0)
        
        return denormalized
    
    def denormalize_tensor(
        self, 
        precip_tensor: torch.Tensor,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        time_coords: Optional[np.ndarray] = None
    ) -> xr.Dataset:
        """Denormalize a PyTorch tensor of precipitation predictions.
        
        This is a convenience method for denormalizing model outputs during
        inference. It converts a PyTorch tensor to an xarray Dataset and
        applies denormalization specifically for precipitation.
        
        Args:
            precip_tensor: Normalized precipitation tensor with shape:
                          - [H, W] for single prediction
                          - [B, H, W] for batch of predictions
                          - [B, 1, H, W] for batch with channel dimension
            lat_coords: Latitude coordinates for spatial dimensions.
            lon_coords: Longitude coordinates for spatial dimensions.
            time_coords: Optional time coordinates for batch dimension.
            
        Returns:
            Denormalized xarray Dataset with precipitation in original units (mm).
            
        Raises:
            RuntimeError: If normalizer has not been fitted.
            ValueError: If tensor shape doesn't match coordinate dimensions.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ConvLSTMNormalizer must be fitted before denormalizing. "
                "Call fit() with training data first."
            )
        
        # Convert tensor to numpy
        precip_array = precip_tensor.detach().cpu().numpy()
        
        # Handle different tensor shapes
        if precip_array.ndim == 2:
            # Single prediction [H, W]
            dims = ["lat", "lon"]
            coords = {"lat": lat_coords, "lon": lon_coords}
        elif precip_array.ndim == 3:
            # Batch of predictions [B, H, W]
            if time_coords is None:
                time_coords = np.arange(precip_array.shape[0])
            dims = ["time", "lat", "lon"]
            coords = {"time": time_coords, "lat": lat_coords, "lon": lon_coords}
        elif precip_array.ndim == 4:
            # Batch with channel dimension [B, 1, H, W]
            if precip_array.shape[1] != 1:
                raise ValueError(
                    f"Expected channel dimension to be 1, got {precip_array.shape[1]}"
                )
            # Squeeze channel dimension
            precip_array = precip_array.squeeze(1)
            if time_coords is None:
                time_coords = np.arange(precip_array.shape[0])
            dims = ["time", "lat", "lon"]
            coords = {"time": time_coords, "lat": lat_coords, "lon": lon_coords}
        else:
            raise ValueError(
                f"Unexpected tensor shape: {precip_array.shape}. "
                f"Expected [H, W], [B, H, W], or [B, 1, H, W]"
            )
        
        # Validate spatial dimensions
        expected_shape = (len(lat_coords), len(lon_coords))
        actual_shape = precip_array.shape[-2:]
        
        if actual_shape != expected_shape:
            raise ValueError(
                f"Tensor spatial dimensions {actual_shape} don't match "
                f"coordinate dimensions {expected_shape}"
            )
        
        # Get precipitation normalization statistics (scalar values, no level dimension)
        precip_mean = float(self.mean["precipitation"].values)
        precip_std = float(self.std["precipitation"].values)
        
        # Manually denormalize precipitation only
        # Step 1: Reverse Z-score normalization
        denormalized_array = precip_array * precip_std + precip_mean
        
        # Step 2: Reverse log1p transformation
        denormalized_array = np.expm1(denormalized_array)
        
        # Step 3: Ensure non-negative (clip small negative values from numerical errors)
        denormalized_array = np.clip(denormalized_array, 0.0, None)
        
        # Create xarray Dataset with denormalized precipitation
        denormalized_ds = xr.Dataset(
            {"precipitation": (dims, denormalized_array)},
            coords=coords
        )
        
        return denormalized_ds
    
    def save(self, filepath: str) -> None:
        """Save normalization statistics to file.
        
        Saves mean and std statistics as a pickle file for later loading.
        
        Args:
            filepath: Path to save statistics (e.g., 'normalizer.pkl').
            
        Raises:
            RuntimeError: If normalizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save unfitted ConvLSTMNormalizer. "
                "Call fit() with training data first."
            )
        
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'std': self.std,
            }, f)
        
        logger.info(f"Normalization statistics saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load normalization statistics from file.
        
        Loads mean and std statistics from a pickle file saved by save().
        
        Args:
            filepath: Path to load statistics from.
            
        Raises:
            FileNotFoundError: If filepath doesn't exist.
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            stats = pickle.load(f)
        
        self.mean = stats['mean']
        self.std = stats['std']
        self._is_fitted = True
        
        logger.info(f"Normalization statistics loaded from {filepath}")
        logger.info(f"Variables: {list(self.mean.data_vars)}")


def stack_channels(
    data: xr.Dataset,
    time_idx: Optional[int] = None,
    time_slice: Optional[slice] = None
) -> np.ndarray:
    """Stack HPA variables and precipitation into channels.
    
    Converts multi-level atmospheric data into a channel-stacked format suitable
    for ConvLSTM processing. This avoids expensive 3D convolutions by flattening
    the vertical dimension into channels.
    
    Channel ordering:
        - Channels 0-10: DPT at 11 pressure levels
        - Channels 11-21: GPH at 11 pressure levels
        - Channels 22-32: TEM at 11 pressure levels
        - Channels 33-43: U at 11 pressure levels
        - Channels 44-54: V at 11 pressure levels
        - Channel 55: Precipitation
    
    Args:
        data: xarray Dataset containing HPA variables and precipitation.
              Expected dimensions: (time, level, lat, lon) for HPA vars,
                                  (time, lat, lon) for precipitation.
        time_idx: Optional single time index to extract. If provided, returns
                 data for one timestep with shape (C, H, W).
        time_slice: Optional time slice to extract multiple timesteps.
                   If provided, returns data with shape (T, C, H, W).
    
    Returns:
        Stacked channel array:
        - If time_idx is provided: shape (56, H, W)
        - If time_slice is provided: shape (T, 56, H, W)
        - Otherwise: shape (T, 56, H, W) for all timesteps
        
    Raises:
        ValueError: If data is missing required variables or has invalid dimensions.
        
    Examples:
        >>> # Single timestep
        >>> channels = stack_channels(data, time_idx=0)
        >>> channels.shape
        (56, 200, 280)
        
        >>> # Multiple timesteps
        >>> channels = stack_channels(data, time_slice=slice(0, 6))
        >>> channels.shape
        (6, 56, 200, 280)
    """
    # Validate required variables
    for var in HPA_VARIABLES:
        if var not in data:
            raise ValueError(f"Missing required HPA variable: {var}")
    
    if "precipitation" not in data:
        raise ValueError("Missing required variable: precipitation")
    
    # Determine time selection
    if time_idx is not None and time_slice is not None:
        raise ValueError("Cannot specify both time_idx and time_slice")
    
    if time_idx is not None:
        # Single timestep
        time_sel = time_idx
        is_single_time = True
    elif time_slice is not None:
        # Multiple timesteps
        time_sel = time_slice
        is_single_time = False
    else:
        # All timesteps
        time_sel = slice(None)
        is_single_time = False
    
    # Stack HPA variables across levels
    hpa_channels = []
    
    for var in HPA_VARIABLES:
        var_data = data[var].isel(time=time_sel)
        
        # Validate dimensions
        if is_single_time:
            # Expected shape: (level, lat, lon)
            if var_data.ndim != 3:
                raise ValueError(
                    f"Variable {var} has unexpected dimensions: {var_data.dims}. "
                    f"Expected (level, lat, lon)"
                )
            
            # Stack levels as channels: (level, lat, lon) -> (level, lat, lon)
            var_array = var_data.values
        else:
            # Expected shape: (time, level, lat, lon)
            if var_data.ndim != 4:
                raise ValueError(
                    f"Variable {var} has unexpected dimensions: {var_data.dims}. "
                    f"Expected (time, level, lat, lon)"
                )
            
            # Stack levels as channels: (time, level, lat, lon) -> (time, level, lat, lon)
            var_array = var_data.values
        
        hpa_channels.append(var_array)
    
    # Get precipitation data
    precip_data = data["precipitation"].isel(time=time_sel)
    
    if is_single_time:
        # Expected shape: (lat, lon)
        if precip_data.ndim != 2:
            raise ValueError(
                f"Precipitation has unexpected dimensions: {precip_data.dims}. "
                f"Expected (lat, lon)"
            )
        
        # Add channel dimension: (lat, lon) -> (1, lat, lon)
        precip_array = precip_data.values[np.newaxis, :, :]
    else:
        # Expected shape: (time, lat, lon)
        if precip_data.ndim != 3:
            raise ValueError(
                f"Precipitation has unexpected dimensions: {precip_data.dims}. "
                f"Expected (time, lat, lon)"
            )
        
        # Add channel dimension: (time, lat, lon) -> (time, 1, lat, lon)
        precip_array = precip_data.values[:, np.newaxis, :, :]
    
    # Concatenate all channels
    if is_single_time:
        # Shape: (5 vars × 11 levels, lat, lon) + (1, lat, lon)
        # = (55, lat, lon) + (1, lat, lon) = (56, lat, lon)
        stacked = np.concatenate(hpa_channels + [precip_array], axis=0)
    else:
        # Shape: (time, 5 vars × 11 levels, lat, lon) + (time, 1, lat, lon)
        # = (time, 55, lat, lon) + (time, 1, lat, lon) = (time, 56, lat, lon)
        stacked = np.concatenate(hpa_channels + [precip_array], axis=1)
    
    # Validate output shape
    expected_channels = len(HPA_VARIABLES) * len(PRESSURE_LEVELS) + 1  # 56
    
    if is_single_time:
        if stacked.shape[0] != expected_channels:
            raise ValueError(
                f"Stacked channels has unexpected shape: {stacked.shape}. "
                f"Expected ({expected_channels}, H, W)"
            )
    else:
        if stacked.shape[1] != expected_channels:
            raise ValueError(
                f"Stacked channels has unexpected shape: {stacked.shape}. "
                f"Expected (T, {expected_channels}, H, W)"
            )
    
    return stacked


class ConvLSTMDataset(Dataset):
    """PyTorch Dataset for ConvLSTM training with sliding windows.
    
    This dataset wraps xarray data and generates sliding window samples for
    time series prediction. Supports optional upstream region inclusion for
    comparative experiments.
    
    Attributes:
        data: Normalized xarray Dataset with atmospheric and precipitation data.
        window_size: Number of historical timesteps to use as input.
        target_offset: Number of timesteps ahead to predict (default: 1).
        region_config: Region boundaries for upstream and downstream areas.
        include_upstream: Whether to include upstream region in input.
    """
    
    def __init__(
        self,
        data: xr.Dataset,
        window_size: int,
        region_config: RegionConfig,
        target_offset: int = 1,
        include_upstream: bool = False
    ):
        """Initialize ConvLSTMDataset.
        
        Args:
            data: xarray Dataset with dimensions (time, level, lat, lon).
            window_size: Number of historical timesteps for input (e.g., 6).
            region_config: RegionConfig defining upstream and downstream boundaries.
            target_offset: Number of timesteps ahead to predict (default: 1).
            include_upstream: Whether to concatenate upstream region (default: False).
            
        Raises:
            ValueError: If data has insufficient timesteps or invalid dimensions.
        """
        self.data = data
        self.window_size = window_size
        self.target_offset = target_offset
        self.region_config = region_config
        self.include_upstream = include_upstream
        
        # Validate data has sufficient timesteps
        num_timesteps = len(data.time)
        min_required = window_size + target_offset
        
        if num_timesteps < min_required:
            raise ValueError(
                f"Dataset has {num_timesteps} timesteps, but requires at least "
                f"{min_required} (window_size={window_size} + target_offset={target_offset})"
            )
        
        # Calculate number of valid windows
        self.num_windows = num_timesteps - window_size - target_offset + 1
        
        if self.num_windows <= 0:
            raise ValueError(
                f"No valid windows can be created. Dataset has {num_timesteps} timesteps, "
                f"window_size={window_size}, target_offset={target_offset}"
            )
        
        # Extract region data
        self._extract_regions()
        
        logger.info(
            f"ConvLSTMDataset initialized: {self.num_windows} windows, "
            f"window_size={window_size}, include_upstream={include_upstream}"
        )
    
    def _extract_regions(self):
        """Extract upstream and downstream regions from data.
        
        Selects spatial subsets based on region_config boundaries and stores
        them for efficient access during training.
        """
        # Extract downstream region
        self.downstream_data = self.data.sel(
            lat=slice(self.region_config.downstream_lat_min, 
                     self.region_config.downstream_lat_max),
            lon=slice(self.region_config.downstream_lon_min,
                     self.region_config.downstream_lon_max)
        )
        
        # Validate downstream region is not empty
        if len(self.downstream_data.lat) == 0 or len(self.downstream_data.lon) == 0:
            raise ValueError(
                f"Downstream region is empty. Check region boundaries: "
                f"lat=[{self.region_config.downstream_lat_min}, "
                f"{self.region_config.downstream_lat_max}], "
                f"lon=[{self.region_config.downstream_lon_min}, "
                f"{self.region_config.downstream_lon_max}]"
            )
        
        logger.info(
            f"Downstream region: lat={len(self.downstream_data.lat)}, "
            f"lon={len(self.downstream_data.lon)}"
        )
        
        # Extract upstream region if needed
        if self.include_upstream:
            self.upstream_data = self.data.sel(
                lat=slice(self.region_config.upstream_lat_min,
                         self.region_config.upstream_lat_max),
                lon=slice(self.region_config.upstream_lon_min,
                         self.region_config.upstream_lon_max)
            )
            
            # Validate upstream region is not empty
            if len(self.upstream_data.lat) == 0 or len(self.upstream_data.lon) == 0:
                raise ValueError(
                    f"Upstream region is empty. Check region boundaries: "
                    f"lat=[{self.region_config.upstream_lat_min}, "
                    f"{self.region_config.upstream_lat_max}], "
                    f"lon=[{self.region_config.upstream_lon_min}, "
                    f"{self.region_config.upstream_lon_max}]"
                )
            
            logger.info(
                f"Upstream region: lat={len(self.upstream_data.lat)}, "
                f"lon={len(self.upstream_data.lon)}"
            )
            
            # Validate spatial ordering: upstream should be west of downstream
            if self.region_config.upstream_lon_max > self.region_config.downstream_lon_min:
                logger.warning(
                    f"Upstream region (lon_max={self.region_config.upstream_lon_max}) "
                    f"overlaps with downstream region (lon_min={self.region_config.downstream_lon_min}). "
                    f"This may not represent true upstream influence."
                )
        else:
            self.upstream_data = None
    
    def __len__(self) -> int:
        """Return number of valid sliding windows."""
        return self.num_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one training sample with sliding window.
        
        Args:
            idx: Window index (0 to num_windows - 1).
            
        Returns:
            Tuple of (input_tensor, target_tensor):
            - input_tensor: Input sequence [T, C, H, W]
              - T = window_size (e.g., 6)
              - C = 56 channels
              - H, W = spatial dimensions (concatenated if include_upstream=True)
            - target_tensor: Target precipitation [H_down, W_down]
              - H_down, W_down = downstream region spatial dimensions
              
        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.num_windows} windows"
            )
        
        # Calculate time indices for this window
        start_time = idx
        end_time = idx + self.window_size
        target_time = end_time + self.target_offset - 1
        
        # Create input sequence
        if self.include_upstream:
            # Stack upstream and downstream regions spatially
            # Get upstream window
            upstream_window = stack_channels(
                self.upstream_data,
                time_slice=slice(start_time, end_time)
            )  # Shape: (T, C, H_up, W_up)
            
            # Get downstream window
            downstream_window = stack_channels(
                self.downstream_data,
                time_slice=slice(start_time, end_time)
            )  # Shape: (T, C, H_down, W_down)
            
            # Concatenate along longitude (width) dimension
            # Upstream is west (left), downstream is east (right)
            input_array = np.concatenate(
                [upstream_window, downstream_window],
                axis=3  # Concatenate along width dimension
            )  # Shape: (T, C, H, W_up + W_down)
            
        else:
            # Use only downstream region
            input_array = stack_channels(
                self.downstream_data,
                time_slice=slice(start_time, end_time)
            )  # Shape: (T, C, H_down, W_down)
        
        # Get target (precipitation only, downstream region only)
        target_precip = self.downstream_data["precipitation"].isel(
            time=target_time
        ).values  # Shape: (H_down, W_down)
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_precip).float()
        
        return input_tensor, target_tensor
