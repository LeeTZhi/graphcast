# Regional Weather Prediction System

A Graph Neural Network (GNN) based system for regional precipitation prediction, inspired by GraphCast architecture. This system predicts 12-hour accumulated precipitation in downstream regions using historical atmospheric data from both upstream and downstream regions.

## Overview

The Regional Weather Prediction System uses a simplified GNN approach that works directly on latitude-longitude grids without requiring icosahedral mesh structures. It implements an encoder-processor-decoder architecture to learn spatial relationships between upstream and downstream weather patterns.

### Key Features

- **Regional Focus**: Explicitly models upstream-downstream relationships for targeted regional predictions
- **Graph-Based Architecture**: Uses GNN message passing to capture spatial dependencies
- **Flexible Configuration**: Easily configurable region boundaries and model hyperparameters
- **Complete Pipeline**: Includes data preprocessing, training, inference, and evaluation tools
- **Weighted Loss**: Emphasizes accurate prediction of heavy precipitation events

## Installation

### Requirements

- Python 3.8+
- JAX (with GPU support recommended)
- Haiku
- xarray
- pandas
- numpy
- scipy
- matplotlib

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import jax; import haiku; import xarray; print('Installation successful!')"
```

## Quick Start

### 1. Data Preprocessing

Convert raw text files to NetCDF format:

```bash
python scripts/preprocess_data.py \
    --lat-file data/raw/Lat.txt \
    --lon-file data/raw/Lon.txt \
    --hpa-dir data/raw/HPA \
    --precip-dir data/raw/precipitation \
    --output data/processed/regional_weather.nc \
    --compression 4
```

### 2. Model Training

Train the regional prediction model:

```bash
python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir checkpoints/experiment_1 \
    --latent-size 256 \
    --num-gnn-layers 12 \
    --learning-rate 1e-4 \
    --num-epochs 100
```

### 3. Generate Predictions

#### Single-step prediction:
```bash
python scripts/run_inference.py \
    --data data/processed/regional_weather.nc \
    --checkpoint checkpoints/experiment_1/best_model.pkl \
    --normalizer checkpoints/experiment_1/normalizer.pkl \
    --target-time "2020-01-15 12:00:00" \
    --output predictions/pred_20200115_12.nc
```

#### Multi-step prediction:
```bash
python scripts/run_inference.py \
    --data data/processed/regional_weather.nc \
    --checkpoint checkpoints/experiment_1/best_model.pkl \
    --normalizer checkpoints/experiment_1/normalizer.pkl \
    --initial-time "2020-01-15 00:00:00" \
    --num-steps 10 \
    --output predictions/pred_sequence.nc
```

### 4. Model Evaluation

Evaluate on test set with visualizations:

```bash
python scripts/evaluate_model.py \
    --data data/processed/regional_weather.nc \
    --checkpoint checkpoints/experiment_1/best_model.pkl \
    --normalizer checkpoints/experiment_1/normalizer.pkl \
    --test-start-time "2020-01-01 00:00:00" \
    --test-end-time "2020-12-31 23:59:59" \
    --output-dir evaluation/experiment_1 \
    --visualizations
```

## Data Format

### Input Data Structure

The system expects raw data in the following format:

```
data/raw/
├── Lat.txt                    # Latitude coordinates
├── Lon.txt                    # Longitude coordinates
├── HPA/                       # Atmospheric variables
│   ├── DPT_20100101_00.txt   # Dew point temperature
│   ├── GPH_20100101_00.txt   # Geopotential height
│   ├── TEM_20100101_00.txt   # Temperature
│   ├── U_20100101_00.txt     # Eastward wind
│   ├── V_20100101_00.txt     # Northward wind
│   └── ...
└── precipitation/             # Precipitation data
    ├── 20100101_8-20.txt     # BJT 08:00-20:00 (UTC 12:00)
    ├── 20100101_20-8.txt     # BJT 20:00-08:00 (UTC 00:00)
    └── ...
```

### Processed Data Format

After preprocessing, data is stored in NetCDF format with the following structure:

```
Dimensions:
  - time: N timestamps (12-hour intervals, UTC)
  - level: 11 pressure levels (100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000 hPa)
  - lat: Latitude points (0.25° resolution)
  - lon: Longitude points (0.25° resolution)

Variables:
  - DPT(time, level, lat, lon): Dew point temperature [K]
  - GPH(time, level, lat, lon): Geopotential height [m]
  - TEM(time, level, lat, lon): Temperature [K]
  - U(time, level, lat, lon): Eastward wind [m/s]
  - V(time, level, lat, lon): Northward wind [m/s]
  - precipitation(time, lat, lon): 12h accumulated precipitation [mm]
```

## Configuration

### Region Configuration

Define upstream and downstream regions in your scripts:

```python
from graphcast_regional.config import RegionConfig

region_config = RegionConfig(
    # Downstream region (target prediction area)
    downstream_lat_min=25.0,
    downstream_lat_max=40.0,
    downstream_lon_min=110.0,
    downstream_lon_max=125.0,
    
    # Upstream region (influencing area)
    upstream_lat_min=25.0,
    upstream_lat_max=50.0,
    upstream_lon_min=70.0,
    upstream_lon_max=110.0,
    
    # Graph connectivity
    intra_domain_k_neighbors=8,
    inter_domain_k_neighbors=32,
)
```

### Model Configuration

Configure model architecture:

```python
from graphcast_regional.config import ModelConfig

model_config = ModelConfig(
    latent_size=256,              # Latent dimension
    num_gnn_layers=12,            # Number of GNN layers
    mlp_hidden_size=256,          # MLP hidden size
    mlp_num_hidden_layers=2,      # MLP depth
    use_residual=True,            # Residual connections
    activation="swish",           # Activation function
    use_layer_norm=True,          # Layer normalization
)
```

### Training Configuration

Configure training hyperparameters:

```python
from graphcast_regional.config import TrainingConfig

training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=100,
    gradient_clip_norm=1.0,
    weight_decay=1e-5,
    warmup_steps=1000,
    high_precip_threshold=10.0,   # Threshold for weighted loss
    high_precip_weight=3.0,       # Weight for heavy precipitation
    validation_frequency=500,
    checkpoint_frequency=1000,
    early_stopping_patience=10,
)
```

## Architecture

### Model Components

1. **Encoder**: Transforms 112-channel input features (5 HPA variables × 11 levels × 2 timesteps + 2 precipitation timesteps) to latent space
2. **Processor**: Multiple GNN layers with message passing and residual connections
3. **Decoder**: Transforms latent representations to precipitation predictions (downstream nodes only)

### Graph Structure

- **Nodes**: Grid points in upstream and downstream regions
- **Edges**:
  - Intra-domain: k-NN connections within downstream region
  - Inter-domain: Connections from upstream to downstream boundary nodes

### Input/Output

- **Input**: Atmospheric data at t-12h and t (2 timesteps)
- **Output**: Precipitation prediction at t+12h (12-hour ahead)

## Evaluation Metrics

The system computes the following metrics:

- **MSE Overall**: Mean squared error across all samples
- **MSE by Intensity**: Separate MSE for light (0-10mm), moderate (10-25mm), and heavy (>25mm) precipitation
- **Spatial Correlation**: Pearson correlation between predicted and actual fields
- **Critical Success Index (CSI)**: Binary classification metric at 1mm, 10mm, and 25mm thresholds

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

**Problem**: GPU/CPU runs out of memory during training.

**Solutions**:
- Reduce batch size: `--batch-size 2` or `--batch-size 1`
- Reduce model size: `--latent-size 128 --num-gnn-layers 8`
- Use gradient accumulation (requires code modification)

#### 2. NaN Loss During Training

**Problem**: Loss becomes NaN during training.

**Solutions**:
- Reduce learning rate: `--learning-rate 1e-5`
- Increase gradient clipping: `--gradient-clip-norm 0.5`
- Check data for NaN values in preprocessing step

#### 3. Missing Data Files

**Problem**: FileNotFoundError during preprocessing.

**Solutions**:
- Verify all HPA files exist for each timestamp
- Check precipitation file naming convention (yyyymmdd_8-20.txt or yyyymmdd_20-8.txt)
- Ensure coordinate files (Lat.txt, Lon.txt) are present

#### 4. Slow Training

**Problem**: Training is very slow.

**Solutions**:
- Ensure JAX is using GPU: `python -c "import jax; print(jax.devices())"`
- Install CUDA-enabled JAX: `pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- Reduce validation frequency: `--validation-frequency 1000`

#### 5. Poor Prediction Quality

**Problem**: Model predictions are inaccurate.

**Solutions**:
- Train for more epochs: `--num-epochs 200`
- Increase model capacity: `--latent-size 512 --num-gnn-layers 16`
- Adjust region boundaries to better capture upstream influences
- Increase high precipitation weight: `--high-precip-weight 5.0`

### Data Quality Checks

Before training, verify your data:

```python
import xarray as xr

# Load processed data
data = xr.open_dataset('data/processed/regional_weather.nc')

# Check for NaN values
print("NaN counts:")
for var in data.data_vars:
    nan_count = data[var].isnull().sum().values
    print(f"  {var}: {nan_count}")

# Check value ranges
print("\nValue ranges:")
for var in data.data_vars:
    print(f"  {var}: [{data[var].min().values:.2f}, {data[var].max().values:.2f}]")

# Check temporal coverage
print(f"\nTime range: {data.time.values[0]} to {data.time.values[-1]}")
print(f"Number of timesteps: {len(data.time)}")
```

## Performance Tips

### Training Optimization

1. **Use GPU**: Ensure JAX is configured to use GPU for significant speedup
2. **Batch Size**: Larger batches (if memory allows) can improve training stability
3. **Learning Rate**: Use warmup and cosine decay for better convergence
4. **Early Stopping**: Enable early stopping to avoid overfitting

### Inference Optimization

1. **JIT Compilation**: Model forward pass is JIT-compiled for faster inference
2. **Batch Predictions**: Generate multiple predictions in sequence for efficiency
3. **Data Caching**: Keep dataset in memory when generating multiple predictions

## Citation

If you use this system in your research, please cite:

```bibtex
@software{regional_weather_prediction,
  title={Regional Weather Prediction System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Inspired by Google DeepMind's GraphCast architecture
- Built with JAX, Haiku, and xarray
- Thanks to the open-source community for excellent tools and libraries

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact: your.email@example.com

## Additional Resources

- [Design Document](.kiro/specs/regional-weather-prediction/design.md): Detailed system design
- [Requirements Document](.kiro/specs/regional-weather-prediction/requirements.md): System requirements
- [Implementation Tasks](.kiro/specs/regional-weather-prediction/tasks.md): Development roadmap
