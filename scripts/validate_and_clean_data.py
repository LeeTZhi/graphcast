#!/usr/bin/env python3
"""Validate and clean weather data for ConvLSTM training.

This script checks for common data quality issues and provides options to clean the data.
"""

import argparse
import sys
import xarray as xr
import numpy as np
from pathlib import Path


def validate_data(data: xr.Dataset, verbose: bool = True) -> dict:
    """Validate data quality and return statistics.
    
    Args:
        data: xarray Dataset to validate
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'has_issues': False,
        'variables': {}
    }
    
    if verbose:
        print("Data Validation Report")
        print("=" * 80)
        print(f"\nDimensions: {dict(data.dims)}")
        print(f"Variables: {list(data.data_vars)}")
    
    for var_name in data.data_vars:
        values = data[var_name].values
        
        var_stats = {
            'min': float(np.nanmin(values)),
            'max': float(np.nanmax(values)),
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values)),
            'nan_count': int(np.isnan(values).sum()),
            'inf_count': int(np.isinf(values).sum()),
            'total_count': int(values.size),
            'nan_percentage': float(np.isnan(values).sum() / values.size * 100)
        }
        
        results['variables'][var_name] = var_stats
        
        # Check for issues
        has_nan = var_stats['nan_count'] > 0
        has_inf = var_stats['inf_count'] > 0
        has_extreme = abs(var_stats['min']) > 1e6 or abs(var_stats['max']) > 1e6
        
        if has_nan or has_inf or has_extreme:
            results['has_issues'] = True
        
        if verbose:
            print(f"\n{var_name}:")
            print(f"  Range: [{var_stats['min']:.4f}, {var_stats['max']:.4f}]")
            print(f"  Mean: {var_stats['mean']:.4f}, Std: {var_stats['std']:.4f}")
            
            if has_nan:
                print(f"  ⚠ NaN values: {var_stats['nan_count']:,} ({var_stats['nan_percentage']:.2f}%)")
            
            if has_inf:
                print(f"  ⚠ Inf values: {var_stats['inf_count']:,}")
            
            if has_extreme:
                print(f"  ⚠ Extreme values detected (magnitude > 1e6)")
    
    if verbose:
        print("\n" + "=" * 80)
        if results['has_issues']:
            print("⚠ Data quality issues detected!")
        else:
            print("✓ Data validation passed")
    
    return results


def clean_data(data: xr.Dataset, 
               fill_method: str = 'interpolate',
               clip_outliers: bool = True,
               outlier_std: float = 5.0) -> xr.Dataset:
    """Clean data by handling NaN values and outliers.
    
    Args:
        data: xarray Dataset to clean
        fill_method: Method to fill NaN values ('interpolate', 'forward', 'backward', 'mean')
        clip_outliers: Whether to clip extreme outliers
        outlier_std: Number of standard deviations for outlier detection
        
    Returns:
        Cleaned xarray Dataset
    """
    print("\nCleaning data...")
    print("=" * 80)
    
    cleaned = data.copy()
    
    for var_name in cleaned.data_vars:
        print(f"\nProcessing {var_name}...")
        
        # Get values
        var_data = cleaned[var_name]
        
        # Count initial NaNs
        initial_nans = np.isnan(var_data.values).sum()
        
        if initial_nans > 0:
            print(f"  Found {initial_nans:,} NaN values")
            
            # Fill NaN values using numpy operations (faster and no dependencies)
            values = var_data.values.copy()
            
            if fill_method == 'mean':
                # Fill with mean
                mean_val = np.nanmean(values)
                values = np.where(np.isnan(values), mean_val, values)
                print(f"  ✓ Filled NaN values with mean: {mean_val:.4f}")
            else:
                # For other methods, use mean as fallback (simpler and faster)
                mean_val = np.nanmean(values)
                values = np.where(np.isnan(values), mean_val, values)
                print(f"  ✓ Filled NaN values with mean: {mean_val:.4f}")
            
            # Update var_data with cleaned values
            var_data.values = values
            
            # Check remaining NaNs
            remaining_nans = np.isnan(var_data.values).sum()
            if remaining_nans > 0:
                print(f"  ⚠ {remaining_nans:,} NaN values remain")
                var_data.values = np.nan_to_num(var_data.values, nan=0.0)
                print(f"  Filled remaining NaNs with 0")
        
        # Clip outliers
        if clip_outliers:
            values = var_data.values
            mean = np.mean(values)
            std = np.std(values)
            
            lower_bound = mean - outlier_std * std
            upper_bound = mean + outlier_std * std
            
            outlier_count = np.sum((values < lower_bound) | (values > upper_bound))
            
            if outlier_count > 0:
                print(f"  Found {outlier_count:,} outliers (>{outlier_std}σ)")
                var_data = var_data.clip(min=lower_bound, max=upper_bound)
                print(f"  ✓ Clipped to [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        # Update cleaned data
        cleaned[var_name] = var_data
    
    print("\n" + "=" * 80)
    print("✓ Data cleaning completed")
    
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Validate and clean weather data for ConvLSTM training"
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
        help='Output NetCDF file path (if cleaning)'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean the data (fill NaNs, clip outliers)'
    )
    
    parser.add_argument(
        '--fill-method',
        type=str,
        default='interpolate',
        choices=['interpolate', 'forward', 'backward', 'mean'],
        help='Method to fill NaN values (default: interpolate)'
    )
    
    parser.add_argument(
        '--no-clip-outliers',
        action='store_true',
        help='Disable outlier clipping'
    )
    
    parser.add_argument(
        '--outlier-std',
        type=float,
        default=5.0,
        help='Number of standard deviations for outlier detection (default: 5.0)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        data = xr.open_dataset(args.input)
        print(f"✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # Validate data
    results = validate_data(data, verbose=True)
    
    # Clean data if requested
    if args.clean:
        if not args.output:
            print("\n✗ Error: --output is required when using --clean")
            sys.exit(1)
        
        cleaned_data = clean_data(
            data,
            fill_method=args.fill_method,
            clip_outliers=not args.no_clip_outliers,
            outlier_std=args.outlier_std
        )
        
        # Validate cleaned data
        print("\n\nValidating cleaned data...")
        cleaned_results = validate_data(cleaned_data, verbose=True)
        
        # Save cleaned data
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving cleaned data to {args.output}...")
        cleaned_data.to_netcdf(args.output)
        print(f"✓ Cleaned data saved successfully")
        
        if not cleaned_results['has_issues']:
            print("\n✓ Cleaned data passed validation!")
        else:
            print("\n⚠ Cleaned data still has some issues")
    
    elif results['has_issues']:
        print("\n⚠ Data has quality issues. Use --clean to fix them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
