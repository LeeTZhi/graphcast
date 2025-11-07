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
"""Data ETL module for loading and processing regional atmospheric data."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import xarray as xr

from graphcast_regional.types import PRESSURE_LEVELS, HPA_VARIABLES


logger = logging.getLogger(__name__)


class DataETL:
    """Handles data extraction, transformation, and loading for regional weather data.
    
    This class processes raw text files containing atmospheric data and precipitation
    into a unified xarray Dataset suitable for model training and inference.
    """
    
    def __init__(
        self,
        lat_file: str,
        lon_file: str,
        hpa_dir: str,
        precip_dir: str
    ):
        """Initialize DataETL with file paths.
        
        Args:
            lat_file: Path to latitude coordinate file.
            lon_file: Path to longitude coordinate file.
            hpa_dir: Directory containing HPA variable files.
            precip_dir: Directory containing precipitation files.
        """
        self.lat_file = Path(lat_file)
        self.lon_file = Path(lon_file)
        self.hpa_dir = Path(hpa_dir)
        self.precip_dir = Path(precip_dir)
        
        # Validate paths exist
        if not self.lat_file.exists():
            raise FileNotFoundError(f"Latitude file not found: {lat_file}")
        if not self.lon_file.exists():
            raise FileNotFoundError(f"Longitude file not found: {lon_file}")
        if not self.hpa_dir.exists():
            raise FileNotFoundError(f"HPA directory not found: {hpa_dir}")
        if not self.precip_dir.exists():
            raise FileNotFoundError(f"Precipitation directory not found: {precip_dir}")
        
        # Cache for loaded data
        self._lat_coords: Optional[np.ndarray] = None
        self._lon_coords: Optional[np.ndarray] = None
        self._timestamps: Optional[List[pd.Timestamp]] = None

    def load_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and extract 1D lat/lon coordinates from files.
        
        Reads coordinate files and extracts 1D arrays. Validates that coordinates
        are properly formatted and within expected ranges.
        
        Returns:
            Tuple of (lat_coords, lon_coords) as 1D numpy arrays.
            
        Raises:
            ValueError: If coordinates are invalid or improperly formatted.
        """
        if self._lat_coords is not None and self._lon_coords is not None:
            return self._lat_coords, self._lon_coords
        
        # Load latitude coordinates
        try:
            # Try loading with comma delimiter first, then fall back to whitespace
            try:
                lat_data = np.loadtxt(self.lat_file, dtype=np.float32, delimiter=',')
            except (ValueError, TypeError):
                lat_data = np.loadtxt(self.lat_file, dtype=np.float32)
            
            # Extract 1D array - handle both 1D and 2D cases
            if lat_data.ndim == 2:
                # If 2D grid format (lat x lon), extract unique latitude values
                # Each row should have the same latitude value repeated
                lat_coords = lat_data[:, 0]  # Take first column
                # Remove duplicates and sort
                lat_coords = np.unique(lat_coords)
            elif lat_data.ndim == 1:
                lat_coords = np.unique(lat_data)
            else:
                raise ValueError(f"Unexpected latitude data shape: {lat_data.shape}")
            
            # Validate latitude range
            if not np.all((lat_coords >= -90) & (lat_coords <= 90)):
                raise ValueError(
                    f"Latitude values out of range [-90, 90]: "
                    f"min={lat_coords.min()}, max={lat_coords.max()}"
                )
            
            logger.info(f"Loaded {len(lat_coords)} latitude coordinates: "
                       f"[{lat_coords.min():.2f}, {lat_coords.max():.2f}]")
            
        except Exception as e:
            raise ValueError(f"Failed to load latitude file {self.lat_file}: {e}")
        
        # Load longitude coordinates
        try:
            # Try loading with comma delimiter first, then fall back to whitespace
            try:
                lon_data = np.loadtxt(self.lon_file, dtype=np.float32, delimiter=',')
            except (ValueError, TypeError):
                lon_data = np.loadtxt(self.lon_file, dtype=np.float32)
            
            # Extract 1D array - handle both 1D and 2D cases
            if lon_data.ndim == 2:
                # If 2D grid format, extract unique longitude values
                # Each column or row should have the same longitude value
                lon_coords = lon_data[0, :]  # Take first row
                # Remove duplicates and sort
                lon_coords = np.unique(lon_coords)
            elif lon_data.ndim == 1:
                lon_coords = np.unique(lon_data)
            else:
                raise ValueError(f"Unexpected longitude data shape: {lon_data.shape}")
            
            # Validate longitude range
            if not np.all((lon_coords >= -180) & (lon_coords <= 360)):
                raise ValueError(
                    f"Longitude values out of range [-180, 360]: "
                    f"min={lon_coords.min()}, max={lon_coords.max()}"
                )
            
            logger.info(f"Loaded {len(lon_coords)} longitude coordinates: "
                       f"[{lon_coords.min():.2f}, {lon_coords.max():.2f}]")
            
        except Exception as e:
            raise ValueError(f"Failed to load longitude file {self.lon_file}: {e}")
        
        # Validate coordinate dimensions match expected grid
        if len(lat_coords) == 0 or len(lon_coords) == 0:
            raise ValueError("Coordinate arrays cannot be empty")
        
        # Cache coordinates
        self._lat_coords = lat_coords
        self._lon_coords = lon_coords
        
        return lat_coords, lon_coords

    def scan_timestamps(self) -> List[pd.Timestamp]:
        """Scan HPA files and build sorted UTC timestamp list.
        
        Scans the HPA directory for files matching the pattern and extracts
        timestamps. Assumes HPA files are named with UTC timestamps.
        
        Returns:
            Sorted list of pandas Timestamps in UTC.
            
        Raises:
            ValueError: If no valid HPA files are found.
        """
        if self._timestamps is not None:
            return self._timestamps
        
        timestamps = set()
        
        # Scan HPA directory for files
        # Support two patterns:
        # 1. VARIABLE_YYYYMMDD_HH.txt (original format)
        # 2. YYYYMMDD_HH_hpaLEVEL_VARIABLE.txt (new format)
        for file_path in self.hpa_dir.glob("*.txt"):
            try:
                # Parse filename to extract timestamp
                filename = file_path.stem
                parts = filename.split("_")
                
                # Try new format first: YYYYMMDD_HH_hpaLEVEL_VARIABLE
                if len(parts) >= 4 and parts[2].startswith("hpa"):
                    date_str = parts[0]
                    hour_str = parts[1]
                    variable = parts[3]
                    
                    # Validate variable name
                    if variable not in HPA_VARIABLES:
                        continue
                
                # Try original format: VARIABLE_YYYYMMDD_HH
                elif len(parts) >= 3:
                    variable = parts[0]
                    date_str = parts[1]
                    hour_str = parts[2]
                    
                    # Validate variable name
                    if variable not in HPA_VARIABLES:
                        continue
                else:
                    continue
                    
                # Parse date and hour (common for both formats)
                if len(date_str) == 8 and len(hour_str) == 2:
                    year = int(date_str[0:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    hour = int(hour_str)
                    
                    # Validate hour (should be 00 or 12 for UTC)
                    if hour not in [0, 12]:
                        logger.warning(f"Unexpected hour {hour} in file {filename}, skipping")
                        continue
                    
                    # Create timestamp
                    timestamp = pd.Timestamp(
                        year=year, month=month, day=day, hour=hour,
                        minute=0, second=0, tz='UTC'
                    )
                    timestamps.add(timestamp)
                        
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse timestamp from file {file_path.name}: {e}")
                continue
        
        if not timestamps:
            raise ValueError(f"No valid HPA files found in {self.hpa_dir}")
        
        # Sort timestamps
        sorted_timestamps = sorted(list(timestamps))
        
        logger.info(f"Found {len(sorted_timestamps)} unique timestamps from "
                   f"{sorted_timestamps[0]} to {sorted_timestamps[-1]}")
        
        # Cache timestamps
        self._timestamps = sorted_timestamps
        
        return sorted_timestamps

    def load_hpa_data(self, timestamp: pd.Timestamp) -> Dict[str, np.ndarray]:
        """Load all HPA variables for a given timestamp.
        
        Loads all 5 HPA variables (DPT, GPH, TEM, U, V) across all 11 pressure
        levels for the specified timestamp. Handles NaN values and validates data.
        
        Args:
            timestamp: UTC timestamp to load data for.
            
        Returns:
            Dictionary mapping variable names to 3D arrays with shape
            (num_levels, num_lat, num_lon).
            
        Raises:
            FileNotFoundError: If required HPA files are missing.
            ValueError: If data dimensions are inconsistent.
        """
        lat_coords, lon_coords = self.load_coordinates()
        expected_shape = (len(PRESSURE_LEVELS), len(lat_coords), len(lon_coords))
        
        hpa_data = {}
        
        # Format timestamp for filename
        date_str = timestamp.strftime("%Y%m%d")
        hour_str = timestamp.strftime("%H")
        
        for variable in HPA_VARIABLES:
            variable_data = []
            
            for level in PRESSURE_LEVELS:
                # Try both filename patterns
                # Pattern 1: VARIABLE_YYYYMMDD_HH.txt (original)
                filename1 = f"{variable}_{date_str}_{hour_str}.txt"
                file_path1 = self.hpa_dir / filename1
                
                # Pattern 2: YYYYMMDD_HH_hpaLEVEL_VARIABLE.txt (new)
                filename2 = f"{date_str}_{hour_str}_hpa{level}_{variable}.txt"
                file_path2 = self.hpa_dir / filename2
                
                # Try pattern 2 first (new format), then pattern 1
                if file_path2.exists():
                    file_path = file_path2
                    filename = filename2
                elif file_path1.exists():
                    file_path = file_path1
                    filename = filename1
                else:
                    raise FileNotFoundError(
                        f"HPA file not found for variable {variable}, "
                        f"level {level}hPa, timestamp {timestamp}. "
                        f"Tried: {filename1} and {filename2}"
                    )
                
                try:
                    # Load data from file - try comma delimiter first
                    try:
                        data = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
                    except (ValueError, TypeError):
                        data = np.loadtxt(file_path, dtype=np.float32)
                    
                    # Validate shape
                    if data.shape != (len(lat_coords), len(lon_coords)):
                        raise ValueError(
                            f"Data shape {data.shape} does not match expected "
                            f"({len(lat_coords)}, {len(lon_coords)}) for {filename}"
                        )
                    
                    # Handle NaN values - replace with interpolated values or zero
                    if np.any(np.isnan(data)):
                        nan_count = np.sum(np.isnan(data))
                        logger.warning(
                            f"Found {nan_count} NaN values in {filename}, "
                            f"replacing with zeros"
                        )
                        data = np.nan_to_num(data, nan=0.0)
                    
                    variable_data.append(data)
                    
                except Exception as e:
                    raise ValueError(f"Failed to load {file_path}: {e}")
            
            # Stack all levels for this variable
            hpa_data[variable] = np.stack(variable_data, axis=0)
            
            # Validate final shape
            if hpa_data[variable].shape != expected_shape:
                raise ValueError(
                    f"Variable {variable} has shape {hpa_data[variable].shape}, "
                    f"expected {expected_shape}"
                )
        
        logger.debug(f"Loaded HPA data for {timestamp}: {list(hpa_data.keys())}")
        
        return hpa_data

    def load_precipitation(self, timestamp: pd.Timestamp) -> Optional[np.ndarray]:
        """Load precipitation data with BJT to UTC time conversion.
        
        Loads precipitation files and converts BJT time ranges to UTC timestamps.
        - yyyymmdd_8-20.txt (BJT 08:00-20:00) maps to yyyymmdd 12:00 UTC
        - yyyymmdd_20-8.txt (BJT 20:00-08:00) maps to yyyymmdd 00:00 UTC
        
        Replaces negative values with zero and handles missing data gracefully.
        
        Args:
            timestamp: UTC timestamp to load precipitation for.
            
        Returns:
            2D numpy array with shape (num_lat, num_lon), or None if file not found.
        """
        lat_coords, lon_coords = self.load_coordinates()
        expected_shape = (len(lat_coords), len(lon_coords))
        
        # Convert UTC timestamp to BJT filename
        # UTC 00:00 -> BJT 08:00 (previous day 20:00 to current day 08:00)
        # UTC 12:00 -> BJT 20:00 (current day 08:00 to 20:00)
        
        if timestamp.hour == 12:
            # UTC 12:00 corresponds to BJT 08:00-20:00 of the same day
            date_str = timestamp.strftime("%Y%m%d")
            time_range = "8-20"
        elif timestamp.hour == 0:
            # UTC 00:00 corresponds to BJT 20:00-08:00 of the same day
            date_str = timestamp.strftime("%Y%m%d")
            time_range = "20-8"
        else:
            logger.warning(
                f"Unexpected hour {timestamp.hour} for precipitation timestamp, "
                f"expected 00 or 12"
            )
            return None
        
        # Construct filename: yyyymmdd_time-range.txt
        filename = f"{date_str}_{time_range}.txt"
        file_path = self.precip_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Precipitation file not found: {file_path}")
            return None
        
        try:
            # Load precipitation data - try comma delimiter first
            try:
                data = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
            except (ValueError, TypeError):
                data = np.loadtxt(file_path, dtype=np.float32)
            
            # Validate shape
            if data.shape != expected_shape:
                raise ValueError(
                    f"Precipitation data shape {data.shape} does not match expected "
                    f"{expected_shape} for {filename}"
                )
            
            # Replace negative values with zero
            negative_count = np.sum(data < 0)
            if negative_count > 0:
                logger.debug(
                    f"Replacing {negative_count} negative precipitation values "
                    f"with zero in {filename}"
                )
                data = np.maximum(data, 0.0)
            
            # Handle NaN values
            if np.any(np.isnan(data)):
                nan_count = np.sum(np.isnan(data))
                logger.warning(
                    f"Found {nan_count} NaN values in {filename}, "
                    f"replacing with zeros"
                )
                data = np.nan_to_num(data, nan=0.0)
            
            logger.debug(f"Loaded precipitation for {timestamp} from {filename}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load precipitation from {file_path}: {e}")
            return None

    def build_dataset(self) -> xr.Dataset:
        """Build complete xarray dataset from all data sources.
        
        Combines coordinate data, HPA variables, and precipitation into a single
        xarray Dataset with proper dimensions and coordinates.
        
        Returns:
            xarray.Dataset with dimensions (time, level, lat, lon) for HPA variables
            and (time, lat, lon) for precipitation.
            
        Raises:
            ValueError: If data loading fails or dimensions are inconsistent.
        """
        logger.info("Building xarray dataset...")
        
        # Load coordinates
        lat_coords, lon_coords = self.load_coordinates()
        
        # Get timestamps
        timestamps = self.scan_timestamps()
        
        # Initialize data arrays
        num_times = len(timestamps)
        num_levels = len(PRESSURE_LEVELS)
        num_lat = len(lat_coords)
        num_lon = len(lon_coords)
        
        # Create empty arrays for all variables
        hpa_arrays = {
            var: np.zeros((num_times, num_levels, num_lat, num_lon), dtype=np.float32)
            for var in HPA_VARIABLES
        }
        precip_array = np.zeros((num_times, num_lat, num_lon), dtype=np.float32)
        
        # Load data for each timestamp
        for t_idx, timestamp in enumerate(timestamps):
            logger.info(f"Loading data for timestamp {t_idx + 1}/{num_times}: {timestamp}")
            
            try:
                # Load HPA data
                hpa_data = self.load_hpa_data(timestamp)
                for var in HPA_VARIABLES:
                    hpa_arrays[var][t_idx] = hpa_data[var]
                
                # Load precipitation data
                precip_data = self.load_precipitation(timestamp)
                if precip_data is not None:
                    precip_array[t_idx] = precip_data
                else:
                    logger.warning(f"No precipitation data for {timestamp}, using zeros")
                    precip_array[t_idx] = 0.0
                    
            except Exception as e:
                logger.error(f"Failed to load data for {timestamp}: {e}")
                raise
        
        # Create xarray Dataset
        data_vars = {}
        
        # Add HPA variables
        for var in HPA_VARIABLES:
            data_vars[var] = (
                ["time", "level", "lat", "lon"],
                hpa_arrays[var],
                {
                    "long_name": self._get_variable_long_name(var),
                    "units": self._get_variable_units(var)
                }
            )
        
        # Add precipitation
        data_vars["precipitation"] = (
            ["time", "lat", "lon"],
            precip_array,
            {
                "long_name": "12-hour accumulated precipitation",
                "units": "mm"
            }
        )
        
        # Convert timestamps to numpy datetime64 for NetCDF compatibility
        # Remove timezone info as NetCDF doesn't support it well
        time_values = pd.DatetimeIndex([t.tz_localize(None) for t in timestamps])
        
        # Create dataset
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": time_values,
                "level": PRESSURE_LEVELS,
                "lat": lat_coords,
                "lon": lon_coords
            },
            attrs={
                "description": "Regional atmospheric and precipitation data",
                "resolution": "0.25 degrees",
                "time_info": "UTC timestamps (timezone-naive), 12-hour intervals",
                "pressure_levels": "hPa",
                "created_by": "DataETL"
            }
        )
        
        logger.info(f"Dataset created successfully with shape: {dataset.dims}")
        
        return dataset
    
    def _get_variable_long_name(self, var: str) -> str:
        """Get long name for variable."""
        names = {
            "DPT": "Dew point temperature",
            "GPH": "Geopotential height",
            "TEM": "Temperature",
            "U": "Eastward wind component",
            "V": "Northward wind component"
        }
        return names.get(var, var)
    
    def _get_variable_units(self, var: str) -> str:
        """Get units for variable."""
        units = {
            "DPT": "K",
            "GPH": "m",
            "TEM": "K",
            "U": "m/s",
            "V": "m/s"
        }
        return units.get(var, "")

    def save_netcdf(self, dataset: xr.Dataset, output_path: str, compression_level: int = 4):
        """Save dataset to NetCDF file with compression.
        
        Args:
            dataset: xarray Dataset to save.
            output_path: Path to output NetCDF file.
            compression_level: Compression level (0-9, higher = more compression).
        """
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up compression encoding for all variables
        encoding = {}
        for var in dataset.data_vars:
            encoding[var] = {
                "zlib": True,
                "complevel": compression_level,
                "dtype": "float32"
            }
        
        logger.info(f"Saving dataset to {output_path} with compression level {compression_level}")
        
        try:
            # Try NETCDF4 first, fall back to NETCDF3_64BIT if not available
            try:
                dataset.to_netcdf(
                    output_path,
                    encoding=encoding,
                    format="NETCDF4"
                )
            except (ImportError, ValueError) as e:
                logger.warning(f"NETCDF4 format not available ({e}), using NETCDF3_64BIT format")
                # NETCDF3 doesn't support compression, so remove zlib encoding
                # Also need to handle time coordinate encoding
                encoding_no_compression = {
                    var: {"dtype": "float32"} for var in dataset.data_vars
                }
                # Add time encoding to handle datetime64
                encoding_no_compression["time"] = {"dtype": "float64", "units": "hours since 1970-01-01"}
                
                dataset.to_netcdf(
                    output_path,
                    encoding=encoding_no_compression,
                    format="NETCDF3_64BIT"
                )
            
            # Log file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Dataset saved successfully ({file_size_mb:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save dataset to {output_path}: {e}")
            raise
