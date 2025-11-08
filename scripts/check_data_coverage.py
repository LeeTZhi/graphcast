#!/usr/bin/env python3
"""Check data coverage and identify missing timesteps."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_netcdf_coverage(netcdf_path: str):
    """Check timestep coverage in processed NetCDF file."""
    print(f"\n{'='*80}")
    print(f"Checking NetCDF file: {netcdf_path}")
    print(f"{'='*80}\n")
    
    # Load dataset
    ds = xr.open_dataset(netcdf_path)
    
    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    
    print(f"Time range: {times.min()} to {times.max()}")
    print(f"Total timesteps: {len(times)}")
    print(f"Duration: {(times.max() - times.min()).days} days")
    
    # Calculate expected timesteps (2 per day)
    expected_days = (times.max() - times.min()).days + 1
    expected_timesteps = expected_days * 2
    print(f"\nExpected timesteps (2 per day): {expected_timesteps}")
    print(f"Actual timesteps: {len(times)}")
    print(f"Missing timesteps: {expected_timesteps - len(times)}")
    print(f"Coverage: {len(times) / expected_timesteps * 100:.1f}%")
    
    # Check time intervals
    time_diffs = times[1:] - times[:-1]
    unique_intervals = time_diffs.unique()
    
    print(f"\nTime intervals found:")
    for interval in unique_intervals:
        count = (time_diffs == interval).sum()
        print(f"  {interval}: {count} occurrences")
    
    # Find gaps (intervals > 12 hours)
    gaps = time_diffs[time_diffs > pd.Timedelta(hours=12)]
    if len(gaps) > 0:
        print(f"\nFound {len(gaps)} gaps (> 12 hours):")
        gap_indices = [i for i, diff in enumerate(time_diffs) if diff > pd.Timedelta(hours=12)]
        for idx in gap_indices[:10]:  # Show first 10 gaps
            print(f"  Gap between {times[idx]} and {times[idx+1]}: {time_diffs[idx]}")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more gaps")
    else:
        print("\nNo gaps found - data is continuous!")
    
    # Check for duplicate timestamps
    duplicates = times[times.duplicated()]
    if len(duplicates) > 0:
        print(f"\nWarning: Found {len(duplicates)} duplicate timestamps!")
    
    ds.close()


def check_raw_data_coverage(hpa_dir: str):
    """Check timestep coverage in raw HPA files."""
    print(f"\n{'='*80}")
    print(f"Checking raw HPA files in: {hpa_dir}")
    print(f"{'='*80}\n")
    
    hpa_path = Path(hpa_dir)
    if not hpa_path.exists():
        print(f"Error: Directory not found: {hpa_dir}")
        return
    
    # Scan for files and extract timestamps
    timestamps = set()
    
    for file_path in hpa_path.glob("*.txt"):
        filename = file_path.stem
        parts = filename.split("_")
        
        try:
            # Try new format: YYYYMMDD_HH_hpaLEVEL_VARIABLE
            if len(parts) >= 4 and parts[2].startswith("hpa"):
                date_str = parts[0]
                hour_str = parts[1]
            # Try original format: VARIABLE_YYYYMMDD_HH
            elif len(parts) >= 3:
                date_str = parts[1]
                hour_str = parts[2]
            else:
                continue
            
            if len(date_str) == 8 and len(hour_str) == 2:
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                hour = int(hour_str)
                
                if hour in [0, 12]:
                    timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour)
                    timestamps.add(timestamp)
        except (ValueError, IndexError):
            continue
    
    if not timestamps:
        print("No valid timestamps found in HPA files!")
        return
    
    times = sorted(list(timestamps))
    times = pd.DatetimeIndex(times)
    
    print(f"Time range: {times.min()} to {times.max()}")
    print(f"Total timesteps: {len(times)}")
    print(f"Duration: {(times.max() - times.min()).days} days")
    
    # Calculate expected timesteps
    expected_days = (times.max() - times.min()).days + 1
    expected_timesteps = expected_days * 2
    print(f"\nExpected timesteps (2 per day): {expected_timesteps}")
    print(f"Actual timesteps: {len(times)}")
    print(f"Missing timesteps: {expected_timesteps - len(times)}")
    print(f"Coverage: {len(times) / expected_timesteps * 100:.1f}%")
    
    # Check time intervals
    time_diffs = times[1:] - times[:-1]
    unique_intervals = time_diffs.unique()
    
    print(f"\nTime intervals found:")
    for interval in unique_intervals:
        count = (time_diffs == interval).sum()
        print(f"  {interval}: {count} occurrences")
    
    # Find gaps
    gaps = time_diffs[time_diffs > pd.Timedelta(hours=12)]
    if len(gaps) > 0:
        print(f"\nFound {len(gaps)} gaps (> 12 hours):")
        gap_indices = [i for i, diff in enumerate(time_diffs) if diff > pd.Timedelta(hours=12)]
        for idx in gap_indices[:10]:
            print(f"  Gap between {times[idx]} and {times[idx+1]}: {time_diffs[idx]}")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more gaps")


def main():
    parser = argparse.ArgumentParser(description='Check data coverage and identify missing timesteps')
    parser.add_argument('--netcdf', type=str, help='Path to processed NetCDF file')
    parser.add_argument('--hpa-dir', type=str, help='Path to raw HPA directory')
    
    args = parser.parse_args()
    
    if not args.netcdf and not args.hpa_dir:
        print("Error: Please specify either --netcdf or --hpa-dir")
        parser.print_help()
        return 1
    
    if args.netcdf:
        check_netcdf_coverage(args.netcdf)
    
    if args.hpa_dir:
        check_raw_data_coverage(args.hpa_dir)
    
    print(f"\n{'='*80}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
