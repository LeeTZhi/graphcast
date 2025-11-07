#!/bin/bash

set -e

DATA_ROOT_PATH=/Users/ltz/DataCard/TimeSeries/Data/2025_10/Time_weather
PRE_DATA_PATH=$DATA_ROOT_PATH/pre_data/pre
SOUNDING_DATA_PATH=$DATA_ROOT_PATH/souding_data/sounding
LAT_FILE=$DATA_ROOT_PATH/pre_data/Lat
LON_FILE=$DATA_ROOT_PATH/pre_data/Lon

OUTPUT_FILE=$DATA_ROOT_PATH/all_weather_data.nc
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"


python $SCRIPT_DIR/preprocess_data.py \
        --lat-file $LAT_FILE \
        --lon-file $LON_FILE \
        --hpa-dir $SOUNDING_DATA_PATH \
        --precip-dir $PRE_DATA_PATH \
        --output $OUTPUT_FILE \
        --compression 3