#!/usr/bin/env python3
"""Aggressively clean weather data with physical bounds.

This script applies physically reasonable bounds to weather variables.
"""

import argparse
import sys
import xarray as xr
import numpy as np
from pathlib import Path


# Physical bounds for weather variables (reasonable ranges)
PHYSICAL_BOUNDS = {
    'precipitation': (0.0, 1000.0),  # mm (0 to 500mm is extreme but possible)
    'dew_point': (180.0, 330.0),  # K (-93°C to 57°C)
    'geopotential_height': (-1000.0, 35000.0),  # m (Dead Sea to stratosphere)
    'temperature': (180.0, 330.0),  # K (-93°C to 57°C)
    'u_wind': (-150.0, 150.0),  # m/s (jet stream max ~150 m/s)
    'v_wind': (-150.0, 150.0),  # m/s
    # Also support original names
    'DPT': (180.0, 330.0),
    'GPH': (-1000.0, 35000.0),
    'TEM': (180.0, 330.0),
    'U': (-150.0, 150.0),
    'V': (-150.0, 150.0),
}


def clean_data_aggressive(data: xr.Dataset, create_mask: bool = True) -> xr.Dataset:
    """Clean data with physical bounds and spatial interpolation for NaN.
    
    Args:
        data: xarray Dataset to clean
        create_mask: If True, create a validity mask for each variable
        
    Returns:
        Cleaned xarray Dataset with optional validity masks
    """
    print("Aggressive Data Cleaning with Spatial Interpolation")
    print("=" * 80)
    
    cleaned = data.copy()
    
    for var_name in cleaned.data_vars:
        print(f"\nProcessing {var_name}...")
        
        values = cleaned[var_name].values.copy()
        
        # Count initial issues
        initial_nans = np.isnan(values).sum()
        initial_infs = np.isinf(values).sum()
        
        print(f"  Initial NaN count: {initial_nans:,}")
        print(f"  Initial Inf count: {initial_infs:,}")
        print(f"  Initial range: [{np.nanmin(values):.4f}, {np.nanmax(values):.4f}]")
        
        # Create validity mask (1 = valid, 0 = filled/invalid)
        if create_mask:
            validity_mask = (~np.isnan(values)) & (~np.isinf(values))
        
        # Apply physical bounds if available
        if var_name in PHYSICAL_BOUNDS:
            lower, upper = PHYSICAL_BOUNDS[var_name]
            
            # Count out-of-bounds values
            out_of_bounds_mask = (values < lower) | (values > upper)
            out_of_bounds = np.sum(out_of_bounds_mask)
            if out_of_bounds > 0:
                print(f"  Found {out_of_bounds:,} values outside physical bounds [{lower}, {upper}]")
                if create_mask:
                    validity_mask = validity_mask & (~out_of_bounds_mask)
            
            # Clip to physical bounds
            values = np.clip(values, lower, upper)
            print(f"  ✓ Clipped to physical bounds: [{lower}, {upper}]")
        
        # Replace Inf with NaN
        if initial_infs > 0:
            values = np.where(np.isinf(values), np.nan, values)
            print(f"  ✓ Replaced {initial_infs:,} Inf values with NaN")
        
        # Fill NaN with spatial interpolation (better than mean for weather data)
        if np.isnan(values).sum() > 0:
            nan_count = np.isnan(values).sum()
            
            # Try spatial interpolation for each time step
            if len(values.shape) >= 3:  # Has spatial dimensions
                print(f"  Applying spatial interpolation for {nan_count:,} NaN values...")
                values = spatial_interpolate_nans(values)
            else:
                # Fallback to mean for non-spatial data
                mean_val = np.nanmean(values)
                if np.isnan(mean_val):
                    if var_name in PHYSICAL_BOUNDS:
                        lower, upper = PHYSICAL_BOUNDS[var_name]
                        mean_val = (lower + upper) / 2
                    else:
                        mean_val = 0.0
                    print(f"  ⚠ All values were NaN, using fallback: {mean_val:.4f}")
                values = np.where(np.isnan(values), mean_val, values)
            
            print(f"  ✓ Filled {nan_count:,} NaN values")
        
        # Final check
        final_nans = np.isnan(values).sum()
        final_infs = np.isinf(values).sum()
        
        if final_nans > 0 or final_infs > 0:
            print(f"  ⚠ WARNING: {final_nans} NaNs and {final_infs} Infs remain!")
            # Force clean
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  ✓ Force cleaned remaining invalid values")
        
        print(f"  Final range: [{np.min(values):.4f}, {np.max(values):.4f}]")
        print(f"  Final mean: {np.mean(values):.4f}, std: {np.std(values):.4f}")
        
        # Update cleaned data
        cleaned[var_name].values = values
        
        # Add validity mask as a separate variable
        if create_mask:
            mask_name = f"{var_name}_mask"
            cleaned[mask_name] = xr.DataArray(
                validity_mask.astype(np.float32),
                dims=cleaned[var_name].dims,
                coords=cleaned[var_name].coords
            )
            valid_ratio = validity_mask.sum() / validity_mask.size * 100
            print(f"  ✓ Created validity mask: {valid_ratio:.2f}% valid data")
    
    print("\n" + "=" * 80)
    print("✓ Aggressive cleaning completed")
    
    return cleaned


def spatial_interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Fill NaN values using spatial interpolation (nearest neighbor).
    
    This is better than mean filling for weather data as it preserves
    spatial patterns and gradients.
    
    Args:
        values: Array with shape (..., H, W) where last 2 dims are spatial
        
    Returns:
        Array with NaNs filled by spatial interpolation
    """
    from scipy.ndimage import distance_transform_edt
    
    result = values.copy()
    
    # Handle different array shapes
    if len(values.shape) == 2:  # (H, W)
        result = _interpolate_2d(result)
    elif len(values.shape) == 3:  # (T, H, W) or (C, H, W)
        for i in range(values.shape[0]):
            result[i] = _interpolate_2d(result[i])
    elif len(values.shape) == 4:  # (T, C, H, W)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                result[i, j] = _interpolate_2d(result[i, j])
    
    return result


def _interpolate_2d(data: np.ndarray) -> np.ndarray:
    """Interpolate NaN values in 2D array using nearest valid neighbor.
    
    Args:
        data: 2D array (H, W)
        
    Returns:
        2D array with NaNs filled
    """
    if not np.any(np.isnan(data)):
        return data
    
    # If all NaN, return zeros
    if np.all(np.isnan(data)):
        return np.zeros_like(data)
    
    # Create mask of valid values
    mask = ~np.isnan(data)
    
    # Find nearest valid value for each NaN position
    ind = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    
    # Fill NaN with nearest valid value
    result = data[tuple(ind)]
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggressively clean weather data with physical bounds and spatial interpolation"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input NetCDF file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output NetCDF file path'
    )
    
    parser.add_argument(
        '--create-mask',
        action='store_true',
        default=True,
        help='Create validity masks for each variable (default: True)'
    )
    
    parser.add_argument(
        '--no-mask',
        action='store_true',
        help='Do not create validity masks'
    )
    
    args = parser.parse_args()
    
    # Handle mask creation flag
    create_mask = args.create_mask and not args.no_mask
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        data = xr.open_dataset(args.input)
        print(f"✓ Data loaded: {dict(data.dims)}")
        print(f"  Variables: {list(data.data_vars)}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # Clean data
    cleaned_data = clean_data_aggressive(data, create_mask=create_mask)
    
    # Save cleaned data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving cleaned data to {args.output}...")
    cleaned_data.to_netcdf(args.output)
    print(f"✓ Cleaned data saved successfully")
    
    # Final validation
    print("\nFinal Validation:")
    print("=" * 80)
    for var_name in cleaned_data.data_vars:
        values = cleaned_data[var_name].values
        print(f"{var_name}:")
        print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
        print(f"  Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
        print(f"  NaN count: {np.isnan(values).sum()}")
        print(f"  Inf count: {np.isinf(values).sum()}")
    
    print("\n✓ Data cleaning complete!")


if __name__ == "__main__":
    main()
