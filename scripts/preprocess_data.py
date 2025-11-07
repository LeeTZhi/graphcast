#!/usr/bin/env python3
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
"""Data preprocessing script for Regional Weather Prediction System.

This script runs the ETL pipeline to convert raw text files into a structured
NetCDF dataset suitable for model training and inference.

Example usage:
    python scripts/preprocess_data.py \\
        --lat-file data/raw/Lat.txt \\
        --lon-file data/raw/Lon.txt \\
        --hpa-dir data/raw/HPA \\
        --precip-dir data/raw/precipitation \\
        --output data/processed/regional_weather.nc \\
        --compression 4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphcast_regional.data_etl import DataETL


def setup_logging(verbose: bool = False):
    """Configure logging for the script.
    
    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess raw weather data into NetCDF format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input paths
    parser.add_argument(
        '--lat-file',
        type=str,
        required=True,
        help='Path to latitude coordinate file (e.g., Lat.txt)'
    )
    parser.add_argument(
        '--lon-file',
        type=str,
        required=True,
        help='Path to longitude coordinate file (e.g., Lon.txt)'
    )
    parser.add_argument(
        '--hpa-dir',
        type=str,
        required=True,
        help='Directory containing HPA variable files'
    )
    parser.add_argument(
        '--precip-dir',
        type=str,
        required=True,
        help='Directory containing precipitation files'
    )
    
    # Output path
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output NetCDF file'
    )
    
    # Processing options
    parser.add_argument(
        '--compression',
        type=int,
        default=4,
        choices=range(0, 10),
        help='NetCDF compression level (0-9, higher = more compression)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    return parser.parse_args()


def main():
    """Main preprocessing workflow."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Regional Weather Data Preprocessing")
    logger.info("=" * 80)
    
    # Log configuration
    logger.info(f"Input configuration:")
    logger.info(f"  Latitude file: {args.lat_file}")
    logger.info(f"  Longitude file: {args.lon_file}")
    logger.info(f"  HPA directory: {args.hpa_dir}")
    logger.info(f"  Precipitation directory: {args.precip_dir}")
    logger.info(f"Output configuration:")
    logger.info(f"  NetCDF file: {args.output}")
    logger.info(f"  Compression level: {args.compression}")
    logger.info("")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Initialize DataETL
        logger.info("Initializing DataETL...")
        etl = DataETL(
            lat_file=args.lat_file,
            lon_file=args.lon_file,
            hpa_dir=args.hpa_dir,
            precip_dir=args.precip_dir
        )
        
        # Load coordinates
        logger.info("Loading coordinates...")
        lat_coords, lon_coords = etl.load_coordinates()
        logger.info(f"  Latitude range: [{lat_coords.min():.2f}, {lat_coords.max():.2f}] "
                   f"({len(lat_coords)} points)")
        logger.info(f"  Longitude range: [{lon_coords.min():.2f}, {lon_coords.max():.2f}] "
                   f"({len(lon_coords)} points)")
        logger.info("")
        
        # Scan timestamps
        logger.info("Scanning timestamps...")
        timestamps = etl.scan_timestamps()
        logger.info(f"  Found {len(timestamps)} timestamps")
        logger.info(f"  Time range: {timestamps[0]} to {timestamps[-1]}")
        logger.info("")
        
        # Build dataset
        logger.info("Building xarray dataset...")
        logger.info("  This may take several minutes depending on data size...")
        dataset = etl.build_dataset()
        logger.info(f"  Dataset dimensions: {dict(dataset.dims)}")
        logger.info(f"  Data variables: {list(dataset.data_vars.keys())}")
        logger.info("")
        
        # Save to NetCDF
        logger.info("Saving to NetCDF...")
        etl.save_netcdf(dataset, args.output, compression_level=args.compression)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Preprocessing completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Output saved to: {args.output}")
        logger.info("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
