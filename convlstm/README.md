# ConvLSTM Weather Prediction Module

This module implements a ConvLSTM-based U-Net architecture for regional precipitation forecasting using spatio-temporal atmospheric data.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Processing](#data-processing)
- [Training](#training)
- [Inference](#inference)
- [Experiment Setup](#experiment-setup)
- [Memory Optimization](#memory-optimization)
- [Troubleshooting](#troubleshooting)
- [Requirements Validation](#requirements-validation)
- [References](#references)

## Overview

The ConvLSTM module provides:
- **ConvLSTMCell**: A single ConvLSTM cell for processing one timestep of spatial data
- **ConvLSTMUNet**: A full encoder-decoder architecture with skip connections for precipitation prediction
- **Comparative Experiments**: Quantify the influence of upstream weather systems on downstream precipitation

## Architecture

### ConvLSTMCell

The ConvLSTMCell implements a convolutional LSTM cell that processes spatial data while maintaining temporal dependencies. It uses:
- Input-to-state convolutions
- State-to-state convolutions
- Four LSTM gates (input, forget, output, cell) with spatial convolutions

**Key Features:**
- Preserves spatial structure through convolutions
- Processes one timestep at a time
- Returns updated hidden and cell states

### ConvLSTMUNet

The ConvLSTMUNet implements a U-Net architecture with ConvLSTM layers:

```
Input [B, T, C, H, W]
    ↓
Encoder (ConvLSTM + MaxPool)
    ↓
Bottleneck (ConvLSTM at reduced resolution)
    ↓
Decoder (Upsample + Skip Connection + ConvLSTM)
    ↓
Output Head (Conv2d 1x1 + ReLU)
    ↓
Output [B, 1, H, W]
```

**Key Features:**
- Encoder-decoder architecture with skip connections
- Downsampling to increase receptive field
- Upsampling to restore original resolution
- ReLU activation for non-negative precipitation values

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU with at least 8GB VRAM (12GB recommended)
- 16GB system RAM (32GB recommended)

### Dependencies

Install all required dependencies:

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install xarray numpy pandas matplotlib hypothesis pytest

# Or install from requirements.txt
pip install -r requirements.txt
```

For GPU support, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to select the appropriate CUDA version for your system.

### Verify Installation

Check that PyTorch can access your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

## Quick Start

### Training a Model

Train a baseline model (downstream region only):

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --batch-size 4 \
    --num-epochs 50
```

Train with upstream region for comparison:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/with_upstream \
    --include-upstream \
    --batch-size 4 \
    --num-epochs 50
```

### Running Inference

Generate predictions from a trained model:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --data data/test_data.nc \
    --output-dir predictions/baseline \
    --visualize
```

### Comparing Experiments

Compare baseline vs. upstream experiments:

```bash
# Generate predictions for both experiments
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --data data/test_data.nc \
    --output-dir predictions/baseline

python run_inference_convlstm.py \
    --checkpoint checkpoints/with_upstream/best_model.pt \
    --data data/test_data.nc \
    --output-dir predictions/with_upstream

# Evaluate and compare
python convlstm/evaluation.py \
    --predictions-exp1 predictions/baseline/predictions.nc \
    --predictions-exp2 predictions/with_upstream/predictions.nc \
    --ground-truth data/test_data.nc
```

## Usage

### Basic Example

```python
import torch
from convlstm.model import ConvLSTMUNet

# Create model
model = ConvLSTMUNet(
    input_channels=56,      # 5 vars × 11 levels + 1 precip
    hidden_channels=[32, 64],
    output_channels=1,
    kernel_size=3
)

# Create input tensor
# Shape: [batch_size, time_steps, channels, height, width]
batch_size = 4
time_steps = 6  # 3 days at 12-hour intervals
channels = 56
height = 20
width = 20

x = torch.randn(batch_size, time_steps, channels, height, width)

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")  # [4, 1, 20, 20]
```

### Using ConvLSTMCell Directly

```python
import torch
from convlstm.model import ConvLSTMCell

# Create cell
cell = ConvLSTMCell(input_dim=56, hidden_dim=32, kernel_size=3)

# Initialize hidden states
batch_size = 4
height = 20
width = 20
device = torch.device('cpu')

h, c = cell.init_hidden(batch_size, height, width, device)

# Process one timestep
input_tensor = torch.randn(batch_size, 56, height, width)
h_next, c_next = cell(input_tensor, (h, c))

print(f"Hidden state shape: {h_next.shape}")  # [4, 32, 20, 20]
```

## Configuration

The model can be configured with different parameters:

```python
from convlstm.config import ConvLSTMConfig

config = ConvLSTMConfig(
    input_channels=56,
    hidden_channels=[32, 64],
    kernel_size=3,
    learning_rate=1e-3,
    batch_size=4,
    num_epochs=100,
    window_size=6,
    use_amp=True  # Mixed precision training
)
```

## Testing

### Running Tests

To verify the implementation, run the test script:

```bash
python convlstm/test_model_simple.py
```

This will test:
- ConvLSTMCell initialization and forward pass
- ConvLSTMUNet initialization and forward pass
- Output shapes and non-negative values
- Different spatial sizes and hidden channel configurations

### Expected Output

```
============================================================
ConvLSTM Model Tests
============================================================

=== Testing ConvLSTMCell ===
Test 1: Cell initialization...
  ✓ Cell initialized correctly
Test 2: Forward pass shape...
  ✓ Output shapes correct: h=(2, 32, 20, 20), c=(2, 32, 20, 20)
Test 3: Hidden state initialization...
  ✓ Hidden states initialized correctly: h=(2, 32, 20, 20), c=(2, 32, 20, 20)
✓ All ConvLSTMCell tests passed!

=== Testing ConvLSTMUNet ===
Test 1: Model initialization...
  ✓ Model initialized correctly
...
✓ All ConvLSTMUNet tests passed!

============================================================
✓ ALL TESTS PASSED!
============================================================
```

## Input Data Format

The model expects input tensors with shape `[B, T, C, H, W]`:
- **B**: Batch size (e.g., 4)
- **T**: Time steps (e.g., 6 for 3 days at 12-hour intervals)
- **C**: Channels (56 = 5 HPA variables × 11 levels + 1 precipitation)
- **H**: Height (spatial dimension)
- **W**: Width (spatial dimension)

### Variable Names

The normalizer automatically handles different variable naming conventions:

**Supported formats:**
- Standard format: `DPT`, `GPH`, `TEM`, `U`, `V`, `precipitation`
- Descriptive format: `dew_point`, `geopotential_height`, `temperature`, `u_wind`, `v_wind`, `precipitation`

The normalizer will automatically rename variables to the standard format during processing.

### Channel Stacking

The 56 input channels are organized as:
- Channels 0-10: DPT (Dew Point Temperature) at 11 pressure levels
- Channels 11-21: GPH (Geopotential Height) at 11 pressure levels
- Channels 22-32: TEM (Temperature) at 11 pressure levels
- Channels 33-43: U (U-wind component) at 11 pressure levels
- Channels 44-54: V (V-wind component) at 11 pressure levels
- Channel 55: Precipitation

## Output Format

The model outputs precipitation predictions with shape `[B, 1, H, W]`:
- **B**: Batch size
- **1**: Single channel (precipitation)
- **H**: Height (same as input)
- **W**: Width (same as input)

All output values are non-negative due to the ReLU activation.

## Memory Optimization

The architecture is designed to work within 12GB GPU memory constraints:

1. **Channel Stacking**: Flattens 3D atmospheric data into 2D with channels, avoiding expensive 3D convolutions
2. **Mixed Precision Training**: Use `torch.cuda.amp` for automatic mixed precision
3. **Gradient Accumulation**: Simulate larger batch sizes by accumulating gradients
4. **Efficient Architecture**: Uses [32, 64] hidden channels by default

## Requirements Validation

This implementation satisfies the following requirements:

### Requirement 2.1 (ConvLSTMCell)
✅ ConvLSTM cells capture temporal dependencies across the window
✅ Input-to-state and state-to-state convolutions implemented
✅ LSTM gates (input, forget, output, cell) with spatial convolutions

### Requirement 2.2 (Encoder)
✅ Downsampling applied to increase receptive field and reduce memory usage

### Requirement 2.3 (Decoder)
✅ Upsampling with skip connections to restore original spatial resolution

### Requirement 2.4 (Output)
✅ Precipitation predictions with same spatial dimensions as input grid

### Requirement 2.5 (Hidden States)
✅ Zero-initialized hidden and cell states with appropriate dimensions

## Next Steps

After implementing the model architecture, the next steps are:
1. Implement data processing and dataset classes (Task 3)
2. Implement data normalization integration (Task 4)
3. Implement weighted loss function (Task 5)
4. Implement training pipeline (Task 6)

## References

- Design Document: `.kiro/specs/convlstm-weather-prediction/design.md`
- Requirements Document: `.kiro/specs/convlstm-weather-prediction/requirements.md`
- Tasks Document: `.kiro/specs/convlstm-weather-prediction/tasks.md`


## Data Processing

### Channel Stacking Function

The `stack_channels()` function converts multi-level atmospheric data into channel-stacked format suitable for ConvLSTM processing.

**Function Signature:**
```python
def stack_channels(
    data: xr.Dataset,
    time_idx: Optional[int] = None,
    time_slice: Optional[slice] = None
) -> np.ndarray
```

**Channel Ordering (56 total):**
- Channels 0-10: DPT (Dew Point Temperature) at 11 pressure levels
- Channels 11-21: GPH (Geopotential Height) at 11 pressure levels  
- Channels 22-32: TEM (Temperature) at 11 pressure levels
- Channels 33-43: U (Eastward Wind) at 11 pressure levels
- Channels 44-54: V (Northward Wind) at 11 pressure levels
- Channel 55: Precipitation

**Usage:**
```python
from convlstm.data import stack_channels

# Single timestep: returns (56, H, W)
channels = stack_channels(data, time_idx=0)

# Multiple timesteps: returns (T, 56, H, W)
channels = stack_channels(data, time_slice=slice(0, 6))

# All timesteps: returns (T, 56, H, W)
channels = stack_channels(data)
```

### ConvLSTMDataset

PyTorch Dataset for ConvLSTM training with sliding windows and optional upstream region inclusion.

**Class Signature:**
```python
class ConvLSTMDataset(Dataset):
    def __init__(
        self,
        data: xr.Dataset,
        window_size: int,
        region_config: RegionConfig,
        target_offset: int = 1,
        include_upstream: bool = False
    )
```

**Parameters:**
- `data`: xarray Dataset with atmospheric variables (normalized)
- `window_size`: Number of historical timesteps (e.g., 6 for 3 days)
- `region_config`: RegionConfig defining spatial boundaries
- `target_offset`: Timesteps ahead to predict (default: 1 = 12 hours)
- `include_upstream`: Whether to concatenate upstream region (default: False)

**Returns:**
- `input_tensor`: Shape (T, C, H, W) where T=window_size, C=56
- `target_tensor`: Shape (H_down, W_down) - downstream precipitation only

**Example: Baseline Experiment (Downstream Only)**
```python
from convlstm.data import ConvLSTMDataset, RegionConfig
from torch.utils.data import DataLoader

# Define downstream region
region_config = RegionConfig(
    downstream_lat_min=25.0,
    downstream_lat_max=40.0,
    downstream_lon_min=110.0,
    downstream_lon_max=125.0
)

# Create dataset (baseline: downstream only)
dataset = ConvLSTMDataset(
    data=normalized_data,
    window_size=6,
    region_config=region_config,
    include_upstream=False
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# Training loop
for inputs, targets in dataloader:
    # inputs: (B, T, C, H, W) = (4, 6, 56, H_down, W_down)
    # targets: (B, H_down, W_down)
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
```

**Example: Comparison Experiment (Upstream + Downstream)**
```python
# Define both regions
region_config = RegionConfig(
    downstream_lat_min=25.0,
    downstream_lat_max=40.0,
    downstream_lon_min=110.0,
    downstream_lon_max=125.0,
    upstream_lat_min=25.0,
    upstream_lat_max=40.0,
    upstream_lon_min=70.0,
    upstream_lon_max=110.0
)

# Create dataset (comparison: upstream + downstream)
dataset = ConvLSTMDataset(
    data=normalized_data,
    window_size=6,
    region_config=region_config,
    include_upstream=True  # Include upstream region
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
for inputs, targets in dataloader:
    # inputs: (B, T, C, H, W_up + W_down) - concatenated regions
    # targets: (B, H_down, W_down) - downstream only
    predictions = model(inputs)
    
    # Loss is computed only on downstream region
    loss = criterion(predictions, targets)
```

### RegionConfig

Configuration dataclass for defining spatial boundaries.

**Attributes:**
```python
@dataclass
class RegionConfig:
    # Downstream region (target prediction area)
    downstream_lat_min: float = 25.0
    downstream_lat_max: float = 40.0
    downstream_lon_min: float = 110.0
    downstream_lon_max: float = 125.0
    
    # Upstream region (influencing area)
    upstream_lat_min: float = 25.0
    upstream_lat_max: float = 50.0
    upstream_lon_min: float = 70.0
    upstream_lon_max: float = 110.0
```

**Usage:**
```python
from convlstm.data import RegionConfig

# Custom region boundaries
region_config = RegionConfig(
    downstream_lat_min=30.0,
    downstream_lat_max=45.0,
    downstream_lon_min=105.0,
    downstream_lon_max=120.0,
    upstream_lat_min=30.0,
    upstream_lat_max=45.0,
    upstream_lon_min=85.0,
    upstream_lon_max=105.0
)
```

### Sliding Window Generation

The ConvLSTMDataset automatically generates sliding windows from the time series:

```
Time series: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]
Window size: 6
Target offset: 1

Window 0: Input [t0, t1, t2, t3, t4, t5] → Target t6
Window 1: Input [t1, t2, t3, t4, t5, t6] → Target t7
Window 2: Input [t2, t3, t4, t5, t6, t7] → Target t8
Window 3: Input [t3, t4, t5, t6, t7, t8] → Target t9
```

Number of windows = `num_timesteps - window_size - target_offset + 1`

### Spatial Concatenation

When `include_upstream=True`, the dataset concatenates upstream and downstream regions along the longitude (width) dimension:

```
Upstream Region          Downstream Region
[H, W_up]                [H, W_down]
    ↓                         ↓
    └─────── Concatenate ─────┘
              ↓
    [H, W_up + W_down]
```

The target is always the downstream region only, ensuring fair comparison between experiments.

## Data Splitting

### create_train_val_test_split

Split data by time ranges into train/validation/test sets with temporal ordering validation.

**Function Signature:**
```python
def create_train_val_test_split(
    data: xr.Dataset,
    train_end_year: Optional[int] = None,
    val_end_year: Optional[int] = None,
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
    test_start_date: Optional[str] = None,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]
```

**Parameters:**
- `data`: Full dataset with time dimension
- `train_end_year`: Last year (inclusive) for training data (optional)
- `val_end_year`: Last year (inclusive) for validation data (optional)
- `train_ratio`: Fraction of data for training (default: 0.85)
- `val_ratio`: Fraction of data for validation (default: 0.15)
- `test_start_date`: Date string (YYYY-MM-DD) when test set starts (optional)

**Returns:**
- Tuple of (train_data, val_data, test_data)

**Splitting Strategies:**

1. **Ratio-based splitting** (default):
```python
from convlstm.data import create_train_val_test_split

# Split data: 85% train, 15% val, remaining test
train, val, test = create_train_val_test_split(
    data,
    train_ratio=0.85,
    val_ratio=0.15
)
```

2. **Year-based splitting**:
```python
# Split by years: train≤2018, val=2019, test≥2020
train, val, test = create_train_val_test_split(
    data,
    train_end_year=2018,
    val_end_year=2019
)
```

3. **Date-based splitting with test cutoff**:
```python
# Split with explicit test start date
train, val, test = create_train_val_test_split(
    data,
    test_start_date='2020-01-01',
    train_ratio=0.85,
    val_ratio=0.15
)
```

**Temporal Ordering Validation:**

The function validates that:
- Input data has monotonically increasing time coordinates
- Each split maintains temporal ordering
- Splits don't overlap in time
- All original timesteps are preserved

This ensures proper time series prediction where future data doesn't leak into training.

**Example: Complete Training Workflow**
```python
from convlstm.data import (
    create_train_val_test_split,
    ConvLSTMNormalizer,
    ConvLSTMDataset,
    RegionConfig
)
import xarray as xr

# 1. Load full dataset
data = xr.open_dataset('weather_data.nc')

# 2. Split data temporally
train_data, val_data, test_data = create_train_val_test_split(
    data,
    train_ratio=0.7,
    val_ratio=0.15
)

# 3. Fit normalizer on training data only
normalizer = ConvLSTMNormalizer()
normalizer.fit(train_data)
normalizer.save('normalizer.pkl')

# 4. Normalize all splits
train_norm = normalizer.normalize(train_data)
val_norm = normalizer.normalize(val_data)
test_norm = normalizer.normalize(test_data)

# 5. Create datasets
region_config = RegionConfig()

train_dataset = ConvLSTMDataset(
    train_norm, window_size=6, region_config=region_config
)
val_dataset = ConvLSTMDataset(
    val_norm, window_size=6, region_config=region_config
)
test_dataset = ConvLSTMDataset(
    test_norm, window_size=6, region_config=region_config
)

print(f"Train windows: {len(train_dataset)}")
print(f"Val windows: {len(val_dataset)}")
print(f"Test windows: {len(test_dataset)}")
```

## Data Normalization

### ConvLSTMNormalizer

Handles feature normalization for ConvLSTM training and inference with special handling for precipitation.

**Normalization Strategy:**
- **HPA variables** (DPT, GPH, TEM, U, V): Z-score normalization (mean=0, std=1)
- **Precipitation**: log1p transformation followed by Z-score normalization

**Class Signature:**
```python
class ConvLSTMNormalizer:
    def fit(self, train_data: xr.Dataset) -> None
    def normalize(self, data: xr.Dataset) -> xr.Dataset
    def denormalize(self, data: xr.Dataset) -> xr.Dataset
    def denormalize_tensor(
        self, 
        precip_tensor: torch.Tensor,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray,
        time_coords: Optional[np.ndarray] = None
    ) -> xr.Dataset
    def save(self, filepath: str) -> None
    def load(self, filepath: str) -> None
```

**Example: Training Workflow**
```python
from convlstm.data import ConvLSTMNormalizer
import xarray as xr

# Load training data
train_data = xr.open_dataset('train_data.nc')

# Create and fit normalizer
normalizer = ConvLSTMNormalizer()
normalizer.fit(train_data)

# Save normalizer for inference
normalizer.save('normalizer.pkl')

# Normalize training data
train_normalized = normalizer.normalize(train_data)

# Use normalized data for training
dataset = ConvLSTMDataset(train_normalized, window_size=6, region_config=region_config)
```

**Example: Inference Workflow**
```python
from convlstm.data import ConvLSTMNormalizer
import torch

# Load normalizer
normalizer = ConvLSTMNormalizer()
normalizer.load('normalizer.pkl')

# Normalize input data
input_normalized = normalizer.normalize(input_data)

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(input_normalized)

# Denormalize predictions back to original units (mm)
lat_coords = input_data.lat.values
lon_coords = input_data.lon.values
predictions_denorm = normalizer.denormalize_tensor(
    predictions,
    lat_coords,
    lon_coords
)

print(f"Predicted precipitation (mm): {predictions_denorm['precipitation'].values}")
```

**Why log1p for Precipitation?**

Precipitation has a highly skewed distribution with many zero values and occasional extreme values. The log1p transformation:
1. Handles zeros gracefully: `log1p(0) = log(1 + 0) = 0`
2. Compresses large values: `log1p(100) ≈ 4.6`
3. Makes the distribution more Gaussian for better neural network training
4. Is reversed with `expm1`: `expm1(log1p(x)) = x`

**Round-Trip Property:**

The normalizer guarantees that `denormalize(normalize(x)) ≈ x` within numerical precision:

```python
# Original data
original = xr.open_dataset('data.nc')

# Normalize then denormalize
normalizer.fit(original)
normalized = normalizer.normalize(original)
recovered = normalizer.denormalize(normalized)

# Check round-trip accuracy
for var in ['DPT', 'GPH', 'TEM', 'U', 'V', 'precipitation']:
    max_error = np.abs(original[var].values - recovered[var].values).max()
    print(f"{var}: max error = {max_error:.2e}")
```

## Requirements Validation (Data Processing)

This implementation satisfies the following requirements:

### Requirement 1.2 (Channel Stacking)
✅ Stack multi-level atmospheric data into channels (5 variables × 11 levels + 1 precipitation = 56 channels)
✅ Handle both single timestep and multi-timestep inputs

### Requirement 1.3 (Sliding Windows)
✅ Generate sliding windows with configurable window_size
✅ Default window_size=6 (3 days at 12-hour intervals)

### Requirement 1.4 (Normalization)
✅ Apply Z-score normalization to atmospheric variables
✅ Apply log1p transformation to precipitation
✅ Provide inverse transformation (denormalization) for predictions

### Requirement 3.1 (Baseline Experiment)
✅ Support downstream region only (include_upstream=False)

### Requirement 3.2 (Comparison Experiment)
✅ Support spatial concatenation of upstream and downstream regions
✅ Upstream region west of downstream (longitude ordering)

### Requirement 3.5 (Spatial Ordering)
✅ Maintain consistent spatial ordering (upstream west of downstream)
✅ Validate region boundaries during initialization

### Requirement 6.3 (Reuse DataNormalizer)
✅ Adapted DataNormalizer for ConvLSTM's channel-stacked format
✅ Compatible with xarray-based data loading infrastructure

### Requirement 9.1 (Denormalization)
✅ Denormalize outputs to original precipitation units (mm)
✅ Support both xarray Dataset and PyTorch tensor denormalization

## Training

### Training Script

The `train_convlstm.py` script provides a comprehensive CLI for training ConvLSTM models. See `TRAINING_GUIDE.md` for detailed documentation.

**Quick Start:**

```bash
# Train baseline model (downstream only)
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline

# Train with upstream region
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/with_upstream \
    --include-upstream
```

**Key Features:**
- Comprehensive CLI with argument groups for all configuration options
- Support for both baseline and comparative experiments
- Memory optimization (mixed precision, gradient accumulation)
- Automatic checkpoint saving and resumption
- Comprehensive logging to file and console
- Error handling with clear messages

**Configuration Options:**
- Data paths and output directory
- Region boundaries (upstream and downstream)
- Model architecture (hidden channels, kernel size)
- Training hyperparameters (learning rate, batch size, epochs)
- Data processing (window size, train/val split ratios)
- Loss function (precipitation threshold and weights)
- Memory optimization (AMP, gradient accumulation, workers)
- Checkpointing (frequency, early stopping)
- Device selection (auto, cuda, mps, cpu)
  - **MPS support**: GPU acceleration on Apple Silicon Mac (M1/M2/M3)

**Output Files:**
- `training.log`: Detailed training logs
- `best_model.pt`: Best model checkpoint (lowest validation loss)
- `checkpoint_epoch_N.pt`: Periodic checkpoints every 10 epochs
- `normalizer.pkl`: Data normalization statistics

For complete documentation, see `TRAINING_GUIDE.md`.

**Apple Silicon Mac Users**: See `MPS_GUIDE.md` for GPU acceleration on M1/M2/M3 Macs.

### Inference

### Running Inference

After training a model, use the inference script to generate predictions:

```bash
python run_inference_convlstm.py \
    --checkpoint checkpoints/baseline/best_model.pt \
    --data data/test_data.nc \
    --output-dir predictions/baseline \
    --visualize
```

**Key Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--data`: Input data (xarray NetCDF format)
- `--output-dir`: Directory for predictions and visualizations
- `--visualize`: Generate precipitation maps (optional)
- `--batch-size`: Batch size for inference (default: 8)

### Inference Output

The script generates:

1. **Predictions (NetCDF):** `predictions/baseline/predictions.nc`
   ```python
   import xarray as xr
   preds = xr.open_dataset('predictions/baseline/predictions.nc')
   print(preds['precipitation'])  # Predicted precipitation in mm
   ```

2. **Visualizations (PNG):** `predictions/baseline/prediction_map_*.png`
   - Precipitation maps with colorbars
   - One image per timestep
   - Downstream region highlighted

3. **Metadata (JSON):** `predictions/baseline/metadata.json`
   - Model configuration
   - Region boundaries
   - Prediction timestamps

### Programmatic Inference

Use the inference module directly in Python:

```python
from convlstm.inference import load_model, generate_predictions
from convlstm.data import ConvLSTMNormalizer
import xarray as xr
import torch

# Load model and normalizer
model, config, region_config = load_model('checkpoints/baseline/best_model.pt')
normalizer = ConvLSTMNormalizer()
normalizer.load('checkpoints/baseline/normalizer.pkl')

# Load and normalize input data
input_data = xr.open_dataset('data/test_data.nc')
input_normalized = normalizer.normalize(input_data)

# Generate predictions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictions = generate_predictions(
    model=model,
    data=input_normalized,
    window_size=config.window_size,
    region_config=region_config,
    device=device,
    batch_size=8
)

# Denormalize predictions
predictions_denorm = normalizer.denormalize_tensor(
    predictions,
    lat_coords=input_data.lat.values,
    lon_coords=input_data.lon.values,
    time_coords=input_data.time.values[config.window_size:]
)

print(f"Predictions shape: {predictions_denorm['precipitation'].shape}")
print(f"Precipitation range: [{predictions_denorm['precipitation'].min().values:.2f}, "
      f"{predictions_denorm['precipitation'].max().values:.2f}] mm")
```

### Batch Inference

Process multiple files efficiently:

```bash
# Create a batch inference script
for file in data/test_*.nc; do
    basename=$(basename $file .nc)
    python run_inference_convlstm.py \
        --checkpoint checkpoints/baseline/best_model.pt \
        --data $file \
        --output-dir predictions/baseline/$basename \
        --visualize
done
```

### Inference Performance

**Typical Performance (12GB GPU):**
- Single prediction: ~50-100ms
- Batch of 8: ~200-300ms
- 100 timesteps: ~5-10 seconds

**Memory Usage:**
- Model: ~10 MB
- Batch of 8: ~500 MB
- Total: <1 GB

For detailed inference documentation, see `INFERENCE_GUIDE.md` and `INFERENCE_CLI_GUIDE.md`.

## Experiment Setup

The ConvLSTM module supports two types of experiments to quantify the influence of upstream weather systems on downstream precipitation:

### Experiment 1: Baseline (Downstream Only)

**Purpose:** Establish baseline performance using only downstream region data.

**Configuration:**
```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --downstream-lat-min 25.0 \
    --downstream-lat-max 40.0 \
    --downstream-lon-min 110.0 \
    --downstream-lon-max 125.0 \
    --batch-size 4 \
    --num-epochs 100
```

**Input Shape:** `[B, T, C, H_down, W_down]`
- Only downstream region is used as input
- Model learns patterns within the target region

### Experiment 2: Comparison (Upstream + Downstream)

**Purpose:** Demonstrate the value of incorporating upstream atmospheric conditions.

**Configuration:**
```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/with_upstream \
    --include-upstream \
    --downstream-lat-min 25.0 \
    --downstream-lat-max 40.0 \
    --downstream-lon-min 110.0 \
    --downstream-lon-max 125.0 \
    --upstream-lat-min 25.0 \
    --upstream-lat-max 40.0 \
    --upstream-lon-min 70.0 \
    --upstream-lon-max 110.0 \
    --batch-size 4 \
    --num-epochs 100
```

**Input Shape:** `[B, T, C, H, W_up + W_down]`
- Upstream and downstream regions are concatenated along longitude
- Model learns how upstream systems influence downstream precipitation
- Loss is computed only on downstream region for fair comparison

### Comparing Results

After training both experiments, compare their performance:

```bash
# Evaluate both models on the same test data
python convlstm/evaluation.py \
    --predictions-exp1 predictions/baseline/predictions.nc \
    --predictions-exp2 predictions/with_upstream/predictions.nc \
    --ground-truth data/test_data.nc \
    --output-dir evaluation_results
```

**Expected Outcome:**
- Experiment 2 should show lower RMSE and MAE on downstream precipitation
- Improvement quantifies the value of upstream information
- Visualizations show where upstream data helps most (e.g., frontal systems)

### Region Selection Guidelines

**Downstream Region:**
- Should be the target area for precipitation prediction
- Typically 15-20° latitude × 15-20° longitude
- Must have sufficient spatial resolution (at least 10×10 grid points)

**Upstream Region:**
- Should be west of downstream region (prevailing wind direction)
- Same latitude range as downstream for consistency
- Width typically 30-40° longitude to capture approaching systems
- Must not overlap with downstream region

**Example Regions (East Asia):**
```python
# Downstream: Eastern China coast
downstream_lat_min = 25.0
downstream_lat_max = 40.0
downstream_lon_min = 110.0
downstream_lon_max = 125.0

# Upstream: Central/Western China
upstream_lat_min = 25.0
upstream_lat_max = 40.0
upstream_lon_min = 70.0
upstream_lon_max = 110.0
```

## Memory Optimization

The ConvLSTM architecture is designed to work within 12GB GPU memory constraints. Here are strategies to optimize memory usage:

### 1. Mixed Precision Training (Recommended)

Use automatic mixed precision (AMP) to reduce memory by ~40%:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --use-amp  # Enable mixed precision (default: True)
```

**How it works:**
- Forward/backward passes use float16 (half precision)
- Model parameters remain in float32 for stability
- Automatic loss scaling prevents underflow

**Memory savings:** ~40% reduction in activation memory

### 2. Gradient Accumulation

Simulate larger batch sizes without increasing memory:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --batch-size 2 \
    --gradient-accumulation-steps 4  # Effective batch size = 2 × 4 = 8
```

**How it works:**
- Process small batches (e.g., 2) sequentially
- Accumulate gradients without updating parameters
- Update parameters after N accumulation steps
- Equivalent to batch size = batch_size × accumulation_steps

**Memory savings:** Allows training with smaller batch sizes

### 3. Reduce Model Size

Adjust hidden channel dimensions:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --hidden-channels 32 64  # Default: [32, 64]
    # For lower memory: --hidden-channels 16 32
    # For higher capacity: --hidden-channels 64 128
```

**Memory impact:**
- `[16, 32]`: ~2GB peak memory (faster, less capacity)
- `[32, 64]`: ~4GB peak memory (balanced, recommended)
- `[64, 128]`: ~8GB peak memory (slower, more capacity)

### 4. Reduce Spatial Resolution

If your data has high spatial resolution, consider downsampling:

```python
# Downsample data before training
data_downsampled = data.coarsen(lat=2, lon=2, boundary='trim').mean()
```

**Memory savings:** 4× reduction for 2× downsampling in each dimension

### 5. Data Loading Optimization

Optimize DataLoader for GPU utilization:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --num-workers 2  # Parallel data loading (default: 2)
    --pin-memory     # Faster GPU transfer (default: True)
```

**Recommendations:**
- `num_workers=0`: Single-threaded (debugging)
- `num_workers=2`: Balanced (recommended for 12GB GPU)
- `num_workers=4`: More parallelism (if CPU allows)

### 6. Checkpoint Frequency

Reduce checkpoint frequency to save disk I/O:

```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --checkpoint-frequency 2000  # Save every 2000 steps (default: 1000)
```

### Memory Usage Summary

**Typical Configuration (12GB GPU):**
```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --batch-size 4 \
    --hidden-channels 32 64 \
    --use-amp \
    --gradient-accumulation-steps 1 \
    --num-workers 2
```

**Expected Memory Usage:**
- Model parameters: ~10 MB
- Optimizer state (AdamW): ~20 MB
- Activations (batch=4, AMP): ~2-3 GB
- Peak memory: ~4-5 GB
- **Margin:** ~7-8 GB available for larger batches or models

**Low Memory Configuration (8GB GPU):**
```bash
python train_convlstm.py \
    --data data/regional_weather.nc \
    --output-dir checkpoints/baseline \
    --batch-size 2 \
    --hidden-channels 16 32 \
    --use-amp \
    --gradient-accumulation-steps 4 \
    --num-workers 1
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Reduce batch size:
   ```bash
   --batch-size 2  # or even 1
   ```

2. Enable mixed precision (if not already):
   ```bash
   --use-amp
   ```

3. Increase gradient accumulation:
   ```bash
   --batch-size 2 --gradient-accumulation-steps 4
   ```

4. Reduce model size:
   ```bash
   --hidden-channels 16 32
   ```

5. Clear GPU cache before training:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

#### 2. NaN Loss During Training

**Error:**
```
WARNING: Loss is NaN at step X
```

**Causes and Solutions:**

1. **Learning rate too high:**
   ```bash
   --learning-rate 1e-4  # Reduce from default 1e-3
   ```

2. **Gradient explosion:**
   ```bash
   --gradient-clip-norm 0.5  # Reduce from default 1.0
   ```

3. **Invalid data (NaN/Inf in inputs):**
   ```python
   # Check for NaN values in data
   import xarray as xr
   data = xr.open_dataset('data.nc')
   for var in data.data_vars:
       nan_count = data[var].isnull().sum().item()
       print(f"{var}: {nan_count} NaN values")
   ```

4. **Numerical instability in loss:**
   - Check precipitation values are non-negative
   - Verify normalization was applied correctly

#### 3. Slow Training Speed

**Issue:** Training takes too long per epoch

**Solutions:**

1. **Enable data prefetching:**
   ```bash
   --num-workers 2  # or 4 if CPU allows
   --pin-memory
   ```

2. **Use mixed precision:**
   ```bash
   --use-amp  # ~2× speedup
   ```

3. **Reduce validation frequency:**
   ```bash
   --validation-frequency 1000  # Validate less often
   ```

4. **Check GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   ```
   - GPU utilization should be >80%
   - If low, increase `num_workers` or reduce data preprocessing

#### 4. Model Not Learning (Loss Not Decreasing)

**Issue:** Validation loss stays constant or increases

**Solutions:**

1. **Check learning rate:**
   ```bash
   --learning-rate 1e-3  # Try different values: 1e-2, 1e-4
   ```

2. **Verify data normalization:**
   ```python
   # Check normalized data statistics
   normalizer = ConvLSTMNormalizer()
   normalizer.load('normalizer.pkl')
   print(normalizer.means)
   print(normalizer.stds)
   ```

3. **Increase model capacity:**
   ```bash
   --hidden-channels 64 128  # Larger model
   ```

4. **Check data quality:**
   - Verify input and target alignment
   - Check for temporal leakage (future data in inputs)
   - Visualize a few training samples

5. **Adjust loss weights:**
   ```bash
   --high-precip-threshold 5.0  # Lower threshold
   --high-precip-weight 5.0     # Higher weight
   ```

#### 5. Checkpoint Loading Errors

**Error:**
```
RuntimeError: Error loading checkpoint: ...
```

**Solutions:**

1. **Check file integrity:**
   ```bash
   ls -lh checkpoints/baseline/best_model.pt
   ```

2. **Verify PyTorch version compatibility:**
   ```python
   import torch
   checkpoint = torch.load('best_model.pt', map_location='cpu')
   print(checkpoint.keys())
   ```

3. **Use strict=False for partial loading:**
   ```python
   model.load_state_dict(checkpoint['model_state_dict'], strict=False)
   ```

4. **Train from scratch if corrupted:**
   ```bash
   rm checkpoints/baseline/best_model.pt
   python train_convlstm.py ...  # Retrain
   ```

#### 6. Data Loading Errors

**Error:**
```
ValueError: Dataset missing required variables
```

**Solutions:**

1. **Check data format:**
   ```python
   import xarray as xr
   data = xr.open_dataset('data.nc')
   print(data)
   print(data.data_vars)
   print(data.dims)
   ```

2. **Verify required variables:**
   - Required: `DPT`, `GPH`, `TEM`, `U`, `V`, `precipitation`
   - Required dimensions: `time`, `level`, `lat`, `lon`

3. **Check coordinate ranges:**
   ```python
   print(f"Time: {data.time.min().values} to {data.time.max().values}")
   print(f"Lat: {data.lat.min().values} to {data.lat.max().values}")
   print(f"Lon: {data.lon.min().values} to {data.lon.max().values}")
   print(f"Levels: {data.level.values}")
   ```

#### 7. Region Boundary Errors

**Error:**
```
ValueError: Region boundaries outside dataset coordinates
```

**Solutions:**

1. **Check dataset bounds:**
   ```python
   import xarray as xr
   data = xr.open_dataset('data.nc')
   print(f"Lat range: [{data.lat.min().values}, {data.lat.max().values}]")
   print(f"Lon range: [{data.lon.min().values}, {data.lon.max().values}]")
   ```

2. **Adjust region boundaries:**
   ```bash
   # Make sure boundaries are within dataset range
   --downstream-lat-min 25.0 \
   --downstream-lat-max 40.0 \
   --downstream-lon-min 110.0 \
   --downstream-lon-max 125.0
   ```

3. **Verify upstream doesn't overlap downstream:**
   ```python
   # upstream_lon_max should be <= downstream_lon_min
   assert upstream_lon_max <= downstream_lon_min
   ```

#### 8. Insufficient Timesteps

**Error:**
```
ValueError: Dataset has fewer timesteps than required
```

**Solutions:**

1. **Check dataset length:**
   ```python
   import xarray as xr
   data = xr.open_dataset('data.nc')
   print(f"Number of timesteps: {len(data.time)}")
   ```

2. **Reduce window size:**
   ```bash
   --window-size 4  # Reduce from default 6
   ```

3. **Use longer dataset:**
   - Minimum required: `window_size + target_offset`
   - Recommended: At least 100 timesteps for meaningful training

### Getting Help

If you encounter issues not covered here:

1. **Check logs:**
   ```bash
   cat checkpoints/baseline/training.log
   ```

2. **Enable debug logging:**
   ```bash
   python train_convlstm.py ... --log-level DEBUG
   ```

3. **Run tests:**
   ```bash
   pytest convlstm/test_model.py -v
   pytest convlstm/test_model_properties.py -v
   ```

4. **Check GPU status:**
   ```bash
   nvidia-smi
   ```

5. **Verify installation:**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

## Requirements Validation (Training)

This implementation satisfies the following requirements:

#### Requirement 10.1 (CLI Arguments)
✅ Accept arguments for data paths, region boundaries, and model hyperparameters
✅ Organized into logical argument groups for clarity

#### Requirement 10.2 (Experiment Type Flag)
✅ `--include-upstream` flag to enable/disable upstream region inclusion
✅ Support for both baseline and comparative experiments

#### Requirement 10.4 (Error Handling)
✅ Display clear error messages with usage examples
✅ Validate configuration before training
✅ Handle missing files, invalid arguments, and training failures

#### Requirement 10.5 (Help Documentation)
✅ Comprehensive help text with examples
✅ Detailed documentation in TRAINING_GUIDE.md
✅ Organized argument groups for easy navigation
