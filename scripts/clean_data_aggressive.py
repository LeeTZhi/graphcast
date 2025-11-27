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


def clean_data_aggressive(data: xr.Dataset) -> xr.Dataset:
    """Clean data with physical bounds and NaN filling.
    
    Args:
        data: xarray Dataset to clean
        
    Returns:
        Cleaned xarray Dataset
    """
    print("Aggressive Data Cleaning")
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
        
        # Apply physical bounds if available
        if var_name in PHYSICAL_BOUNDS:
            lower, upper = PHYSICAL_BOUNDS[var_name]
            
            # Count out-of-bounds values
            out_of_bounds = np.sum((values < lower) | (values > upper))
            if out_of_bounds > 0:
                print(f"  Found {out_of_bounds:,} values outside physical bounds [{lower}, {upper}]")
            
            # Clip to physical bounds
            values = np.clip(values, lower, upper)
            print(f"  ✓ Clipped to physical bounds: [{lower}, {upper}]")
        
        # Replace Inf with NaN
        if initial_infs > 0:
            values = np.where(np.isinf(values), np.nan, values)
            print(f"  ✓ Replaced {initial_infs:,} Inf values with NaN")
        
        # Fill NaN with mean of valid values
        if np.isnan(values).sum() > 0:
            mean_val = np.nanmean(values)
            if np.isnan(mean_val):
                # If all values are NaN, use middle of physical bounds
                if var_name in PHYSICAL_BOUNDS:
                    lower, upper = PHYSICAL_BOUNDS[var_name]
                    mean_val = (lower + upper) / 2
                else:
                    mean_val = 0.0
                print(f"  ⚠ All values were NaN, using fallback: {mean_val:.4f}")
            
            nan_count = np.isnan(values).sum()
            values = np.where(np.isnan(values), mean_val, values)
            print(f"  ✓ Filled {nan_count:,} NaN values with mean: {mean_val:.4f}")
        
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
    
    print("\n" + "=" * 80)
    print("✓ Aggressive cleaning completed")
    
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Aggressively clean weather data with physical bounds"
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
    
    args = parser.parse_args()
    
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
    cleaned_data = clean_data_aggressive(data)
    
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
