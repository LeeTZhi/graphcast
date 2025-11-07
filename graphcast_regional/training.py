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
"""Training pipeline for Regional Weather Prediction System."""

import os
import pickle
from typing import Dict, Iterator, Optional, Tuple, Any, List
import logging
from functools import partial
import threading
import queue

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr

from graphcast import typed_graph
from graphcast_regional.config import ModelConfig, RegionConfig, TrainingConfig
from graphcast_regional.model import RegionalGNN
from graphcast_regional import types


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataNormalizer:
    """Handles feature normalization for training and inference.
    
    Computes mean and standard deviation from training data and applies
    normalization/denormalization to datasets.
    """
    
    def __init__(self):
        """Initialize DataNormalizer."""
        self.mean: Optional[xr.Dataset] = None
        self.std: Optional[xr.Dataset] = None
        self._is_fitted = False
    
    def fit(self, train_data: xr.Dataset) -> None:
        """Compute normalization statistics from training data.
        
        Args:
            train_data: Training dataset with dimensions (time, level, lat, lon)
                for HPA variables and (time, lat, lon) for precipitation.
        """
        logger.info("Computing normalization statistics from training data...")
        
        # Compute mean and std for each variable
        self.mean = train_data.mean(dim=["time", "lat", "lon"])
        self.std = train_data.std(dim=["time", "lat", "lon"])
        
        # Avoid division by zero - replace zero std with 1.0
        for var_name in self.std.data_vars:
            std_values = self.std[var_name].values
            std_values = np.where(std_values == 0, 1.0, std_values)
            self.std[var_name].values = std_values
        
        self._is_fitted = True
        
        logger.info("Normalization statistics computed successfully")
        logger.info(f"Variables: {list(self.mean.data_vars)}")
    
    def normalize(self, data: xr.Dataset) -> xr.Dataset:
        """Apply normalization to data.
        
        Args:
            data: Dataset to normalize.
            
        Returns:
            Normalized dataset.
            
        Raises:
            RuntimeError: If normalizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("DataNormalizer must be fitted before normalizing data")
        
        normalized = (data - self.mean) / self.std
        return normalized
    
    def denormalize(self, data: xr.Dataset) -> xr.Dataset:
        """Reverse normalization.
        
        Args:
            data: Normalized dataset.
            
        Returns:
            Denormalized dataset.
            
        Raises:
            RuntimeError: If normalizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("DataNormalizer must be fitted before denormalizing data")
        
        denormalized = data * self.std + self.mean
        return denormalized
    
    def save(self, filepath: str) -> None:
        """Save normalization statistics to file.
        
        Args:
            filepath: Path to save statistics.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted DataNormalizer")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'std': self.std,
            }, f)
        
        logger.info(f"Normalization statistics saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load normalization statistics from file.
        
        Args:
            filepath: Path to load statistics from.
        """
        with open(filepath, 'rb') as f:
            stats = pickle.load(f)
        
        self.mean = stats['mean']
        self.std = stats['std']
        self._is_fitted = True
        
        logger.info(f"Normalization statistics loaded from {filepath}")



def create_train_val_test_split(
    data: xr.Dataset,
    train_end_year: Optional[int] = None,
    val_end_year: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_start_date: Optional[str] = None,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Split data by time ranges into train/val/test sets.
    
    Args:
        data: Full dataset with time dimension.
        train_end_year: Last year (inclusive) for training data. If None, uses ratios or dates.
        val_end_year: Last year (inclusive) for validation data. If None, uses ratios or dates.
        train_ratio: Fraction of data for training (if years not specified).
        val_ratio: Fraction of data for validation (if years not specified).
        test_start_date: Date string (YYYY-MM-DD) when test set starts. If specified,
                        data before this date is split into train/val using ratios.
        
    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    logger.info("Splitting data into train/val/test sets...")
    
    # Extract time coordinates
    times = data.time.values
    
    # Log the actual time range in the data
    logger.info(f"Data time range: {times.min()} to {times.max()}")
    logger.info(f"Total timesteps: {len(times)}")
    
    # If test_start_date is specified, use date-based splitting
    if test_start_date is not None:
        logger.info(f"Using date-based split: test starts at {test_start_date}")
        
        test_cutoff = np.datetime64(test_start_date)
        
        # Split into train+val and test
        trainval_mask = times < test_cutoff
        test_mask = times >= test_cutoff
        
        trainval_data = data.isel(time=trainval_mask)
        test_data = data.isel(time=test_mask)
        
        # Further split train+val using ratios
        n_trainval = len(trainval_data.time)
        n_train = int(n_trainval * train_ratio / (train_ratio + val_ratio))
        
        train_data = trainval_data.isel(time=slice(0, n_train))
        val_data = trainval_data.isel(time=slice(n_train, None))
        
        logger.info(f"Train+Val cutoff: {test_cutoff}")
        logger.info(f"Train/Val split ratio: {train_ratio}:{val_ratio}")
        
    # If years are not specified, use ratio-based splitting
    elif train_end_year is None or val_end_year is None:
        logger.info(f"Using ratio-based split: train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio}")
        
        n_total = len(times)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data.isel(time=slice(0, n_train))
        val_data = data.isel(time=slice(n_train, n_train + n_val))
        test_data = data.isel(time=slice(n_train + n_val, None))
    else:
        # Use year-based splitting
        logger.info(f"Using year-based split: train<={train_end_year}, val<={val_end_year}")
        
        # Create time-based masks
        train_mask = times < np.datetime64(f'{train_end_year + 1}-01-01')
        val_mask = (times >= np.datetime64(f'{train_end_year + 1}-01-01')) & \
                   (times < np.datetime64(f'{val_end_year + 1}-01-01'))
        test_mask = times >= np.datetime64(f'{val_end_year + 1}-01-01')
        
        # Split data
        train_data = data.isel(time=train_mask)
        val_data = data.isel(time=val_mask)
        test_data = data.isel(time=test_mask)
    
    logger.info(f"Train samples: {len(train_data.time)}")
    logger.info(f"Val samples: {len(val_data.time)}")
    logger.info(f"Test samples: {len(test_data.time)}")
    
    # Validate that we have training data
    if len(train_data.time) == 0:
        raise ValueError(
            f"No training data found. Data time range is {times.min()} to {times.max()}. "
            f"Please adjust train_end_year and val_end_year to match your data, "
            f"or use ratio-based splitting by setting both to None."
        )
    
    return train_data, val_data, test_data


def create_sliding_windows(
    data: xr.Dataset,
    window_size: int = 2,
    target_offset: int = 1,
    downstream_region: Optional[Tuple[float, float, float, float]] = None,
) -> Iterator[Tuple[xr.Dataset, xr.Dataset]]:
    """Create sliding window (input, target) pairs.
    
    Creates windows where input contains data at time t-12h and t,
    and target contains precipitation at time t+12h.
    
    Args:
        data: Dataset with time dimension.
        window_size: Number of timesteps in input window (default: 2).
        target_offset: Offset from last input timestep to target (default: 1).
        downstream_region: Tuple of (lat_min, lat_max, lon_min, lon_max) for downstream region.
                          If provided, targets are cropped to this region.
        
    Yields:
        Tuple of (input_data, target_data) where:
            - input_data has dimensions (time=window_size, level, lat, lon)
            - target_data has dimensions (lat, lon) for precipitation only (cropped to downstream if specified)
    """
    num_times = len(data.time)
    
    # Need at least window_size + target_offset timesteps
    if num_times < window_size + target_offset:
        logger.warning(
            f"Dataset has only {num_times} timesteps, need at least "
            f"{window_size + target_offset} for windowing"
        )
        return
    
    # Generate windows
    for i in range(num_times - window_size - target_offset + 1):
        # Input: timesteps [i, i+1, ..., i+window_size-1]
        input_indices = slice(i, i + window_size)
        input_data = data.isel(time=input_indices)
        
        # Target: precipitation at timestep i+window_size+target_offset-1
        target_index = i + window_size + target_offset - 1
        target_data = data.isel(time=target_index)["precipitation"]
        
        # Crop target to downstream region if specified
        if downstream_region is not None:
            lat_min, lat_max, lon_min, lon_max = downstream_region
            target_data = target_data.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
        
        # Convert target to Dataset for consistency
        target_dataset = xr.Dataset({
            "precipitation": target_data
        })
        
        yield input_data, target_dataset


class DataPrefetcher:
    """Prefetch data in background thread to keep GPU busy."""
    
    def __init__(self, data_iterator: Iterator, buffer_size: int = 4):
        """Initialize data prefetcher.
        
        Args:
            data_iterator: Iterator that yields (input, target) pairs.
            buffer_size: Number of samples to prefetch.
        """
        self.data_iterator = data_iterator
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.thread = None
        self._stop_event = threading.Event()
    
    def _prefetch_worker(self):
        """Worker function that prefetches data."""
        try:
            for item in self.data_iterator:
                if self._stop_event.is_set():
                    break
                self.queue.put(item)
            self.queue.put(None)  # Signal end of data
        except Exception as e:
            logger.error(f"Error in prefetch worker: {e}")
            self.queue.put(None)
    
    def start(self):
        """Start prefetching in background thread."""
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()
    
    def __iter__(self):
        """Iterate over prefetched data."""
        self.start()
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item
    
    def stop(self):
        """Stop prefetching."""
        self._stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)



class WeightedMSELoss:
    """Weighted MSE loss for precipitation prediction.
    
    Applies higher weights to grid points with precipitation exceeding
    a threshold to improve prediction of heavy precipitation events.
    """
    
    def __init__(
        self,
        high_precip_threshold: float = 10.0,
        high_precip_weight: float = 3.0,
    ):
        """Initialize WeightedMSELoss.
        
        Args:
            high_precip_threshold: Threshold (mm) for high precipitation.
            high_precip_weight: Weight multiplier for high precipitation samples.
        """
        self.high_precip_threshold = high_precip_threshold
        self.high_precip_weight = high_precip_weight
    
    def __call__(
        self,
        predictions: xr.Dataset,
        targets: xr.Dataset,
    ) -> jnp.ndarray:
        """Compute weighted MSE loss.
        
        Args:
            predictions: Predicted precipitation with dimensions (lat, lon).
            targets: Target precipitation with dimensions (lat, lon).
            
        Returns:
            Scalar loss value.
        """
        # Extract precipitation arrays - use .data to get JAX arrays directly
        pred_precip = predictions["precipitation"].data
        target_precip = targets["precipitation"].data
        
        # Compute squared errors
        squared_errors = (pred_precip - target_precip) ** 2
        
        # Create weights based on target precipitation intensity
        weights = jnp.where(
            target_precip > self.high_precip_threshold,
            self.high_precip_weight,
            1.0
        )
        
        # Apply weights
        weighted_errors = squared_errors * weights
        
        # Compute mean weighted loss
        loss = jnp.mean(weighted_errors)
        
        return loss



class TrainingPipeline:
    """Manages model training workflow.
    
    Handles data loading, normalization, training loop, and checkpointing.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        region_config: RegionConfig,
        training_config: TrainingConfig,
        data_path: str,
        graph: typed_graph.TypedGraph,
        output_dir: str = "./checkpoints",
    ):
        """Initialize training pipeline.
        
        Args:
            model_config: Model architecture configuration.
            region_config: Region boundaries configuration.
            training_config: Training hyperparameters.
            data_path: Path to NetCDF dataset.
            graph: Regional graph structure.
            output_dir: Directory for saving checkpoints and logs.
        """
        self.model_config = model_config
        self.region_config = region_config
        self.training_config = training_config
        self.data_path = data_path
        self.graph = graph
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize normalizer
        self.normalizer = DataNormalizer()
        
        # Initialize loss function
        self.loss_fn = WeightedMSELoss(
            high_precip_threshold=training_config.high_precip_threshold,
            high_precip_weight=training_config.high_precip_weight,
        )
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize Haiku model and transform."""
        def forward_fn(inputs: xr.Dataset, is_training: bool = False) -> xr.Dataset:
            model = RegionalGNN(self.model_config, self.region_config)
            return model(inputs, self.graph, is_training=is_training)
        
        self.forward = hk.transform(forward_fn)
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with learning rate schedule.
        
        Returns:
            Optax optimizer.
        """
        # Learning rate schedule with warmup
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.training_config.learning_rate,
            warmup_steps=self.training_config.warmup_steps,
            decay_steps=10000,  # Will be updated based on dataset size
            end_value=self.training_config.learning_rate * 0.1,
        )
        
        # AdamW optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.training_config.gradient_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.training_config.weight_decay,
            ),
        )
        
        return optimizer
    
    def _loss_fn_wrapper(
        self,
        params: hk.Params,
        inputs: xr.Dataset,
        targets: xr.Dataset,
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Wrapper for loss function with model forward pass.
        
        Args:
            params: Model parameters.
            inputs: Input data.
            targets: Target data.
            rng: Random key for dropout, etc.
            
        Returns:
            Scalar loss value.
        """
        # Forward pass
        predictions = self.forward.apply(params, rng, inputs, is_training=True)
        
        # Compute loss
        loss = self.loss_fn(predictions, targets)
        
        return loss
    
    def _train_step(
        self,
        params: hk.Params,
        opt_state: optax.OptState,
        inputs: xr.Dataset,
        targets: xr.Dataset,
        rng: jax.random.PRNGKey,
        optimizer: optax.GradientTransformation,
    ) -> Tuple[hk.Params, optax.OptState, jnp.ndarray]:
        """Single training step with gradient computation and update.
        
        The loss computation and gradient update are JIT compiled internally
        via the Haiku transform and optax optimizer.
        
        Args:
            params: Current model parameters.
            opt_state: Current optimizer state.
            inputs: Input data batch (xarray Dataset).
            targets: Target data batch (xarray Dataset).
            rng: Random key.
            optimizer: Optax optimizer.
            
        Returns:
            Tuple of (updated_params, updated_opt_state, loss_value).
        """
        # Compute loss and gradients
        # Note: The forward pass (self.forward.apply) is already JIT compiled by Haiku
        loss, grads = jax.value_and_grad(self._loss_fn_wrapper)(params, inputs, targets, rng)
        
        # Update parameters (optimizer.update is also JIT compiled by optax)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    def _eval_step(
        self,
        params: hk.Params,
        inputs: xr.Dataset,
        targets: xr.Dataset,
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Single evaluation step.
        
        The loss computation is JIT compiled internally via the Haiku transform.
        
        Args:
            params: Model parameters.
            inputs: Input data (xarray Dataset).
            targets: Target data (xarray Dataset).
            rng: Random key.
            
        Returns:
            Loss value.
        """
        return self._loss_fn_wrapper(params, inputs, targets, rng)
    
    def train(
        self,
        train_end_year: Optional[int] = None,
        val_end_year: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_start_date: Optional[str] = None,
        seed: int = 42,
        use_prefetch: bool = True,
        prefetch_buffer_size: int = 4,
        resume_from: Optional[str] = None,
    ) -> Tuple[hk.Params, DataNormalizer]:
        """Run full training loop with GPU optimizations.
        
        Args:
            train_end_year: Last year (inclusive) for training data. If None, uses ratios.
            val_end_year: Last year (inclusive) for validation data. If None, uses ratios.
            train_ratio: Fraction of data for training (default: 0.7).
            val_ratio: Fraction of data for validation (default: 0.15).
            test_start_date: Date string (YYYY-MM-DD) when test set starts.
            seed: Random seed for initialization.
            use_prefetch: Whether to use data prefetching (default: True).
            prefetch_buffer_size: Number of samples to prefetch (default: 4).
            resume_from: Path to checkpoint file to resume training from (optional).
            
        Returns:
            Tuple of (trained_params, normalizer).
        """
        logger.info("Starting training pipeline with GPU optimizations...")
        logger.info(f"JAX devices: {jax.devices()}")
        logger.info(f"Default backend: {jax.default_backend()}")
        
        # Load data
        logger.info(f"Loading data from {self.data_path}...")
        data = xr.open_dataset(self.data_path)
        
        # Split data
        train_data, val_data, test_data = create_train_val_test_split(
            data, train_end_year, val_end_year, train_ratio, val_ratio, test_start_date
        )
        
        # Fit normalizer on training data
        self.normalizer.fit(train_data)
        
        # Save normalizer
        normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
        self.normalizer.save(normalizer_path)
        
        # Normalize data
        train_data_norm = self.normalizer.normalize(train_data)
        val_data_norm = self.normalizer.normalize(val_data)
        
        # Initialize model parameters
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        
        # Define downstream region for target cropping
        downstream_region = (
            self.region_config.downstream_lat_min,
            self.region_config.downstream_lat_max,
            self.region_config.downstream_lon_min,
            self.region_config.downstream_lon_max,
        )
        
        # Get a sample input for initialization
        sample_windows = list(create_sliding_windows(train_data_norm, downstream_region=downstream_region))
        if not sample_windows:
            raise ValueError("No training windows created - check data size")
        
        sample_input, _ = sample_windows[0]
        
        # Check if resuming from checkpoint
        start_epoch = 0
        start_step = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        if resume_from:
            logger.info(f"Resuming training from checkpoint: {resume_from}")
            checkpoint_data = load_checkpoint(resume_from)
            
            params = checkpoint_data['params']
            opt_state = checkpoint_data.get('opt_state')
            start_epoch = checkpoint_data.get('epoch', 0)
            start_step = checkpoint_data.get('step', 0)
            best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            
            # Move params to GPU
            params = jax.device_put(params, jax.devices()[0])
            
            # Create optimizer and restore state
            optimizer = self._create_optimizer()
            if opt_state is None:
                logger.warning("Optimizer state not found in checkpoint, reinitializing...")
                opt_state = optimizer.init(params)
            
            logger.info(f"Resumed from epoch {start_epoch}, step {start_step}")
            logger.info(f"Best validation loss so far: {best_val_loss:.4f}")
            
            # Try to load existing normalizer
            normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
            if os.path.exists(normalizer_path):
                logger.info(f"Loading existing normalizer from {normalizer_path}")
                self.normalizer.load(normalizer_path)
            else:
                logger.warning("Normalizer not found, refitting on training data...")
                self.normalizer.fit(train_data)
                self.normalizer.save(normalizer_path)
        else:
            logger.info("Initializing model parameters...")
            params = self.forward.init(init_rng, sample_input, is_training=False)
            
            # Move params to GPU
            params = jax.device_put(params, jax.devices()[0])
            
            num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
            logger.info(f"Model initialized with {num_params:,} parameters")
            
            # Create optimizer
            optimizer = self._create_optimizer()
            opt_state = optimizer.init(params)
        
        # Note: JIT compilation happens automatically in Haiku transforms and optax
        logger.info("Model ready for training (JIT compilation will occur on first step)...")
        
        # Training loop
        step = start_step
        
        for epoch in range(start_epoch, self.training_config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
            
            # Training
            epoch_losses = []
            
            # Create data iterator
            data_iter = create_sliding_windows(train_data_norm, downstream_region=downstream_region)
            
            # Optionally wrap with prefetcher
            if use_prefetch:
                data_iter = DataPrefetcher(data_iter, buffer_size=prefetch_buffer_size)
            
            for input_data, target_data in data_iter:
                rng, step_rng = jax.random.split(rng)
                
                # Training step
                # Note: The forward pass and optimizer updates are JIT compiled internally
                params, opt_state, loss = self._train_step(
                    params, opt_state, input_data, target_data,
                    step_rng, optimizer
                )
                
                # Block until computation is done and get loss value
                loss_value = float(jax.device_get(loss))
                epoch_losses.append(loss_value)
                step += 1
                
                # Validation
                if step % self.training_config.validation_frequency == 0:
                    val_loss = self._validate(params, val_data_norm, rng)
                    logger.info(
                        f"Step {step}: train_loss={np.mean(epoch_losses[-100:]):.4f}, "
                        f"val_loss={val_loss:.4f}"
                    )
                    
                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best model with full state
                        self._save_checkpoint(
                            params, "best_model.pkl",
                            opt_state=opt_state,
                            epoch=epoch,
                            step=step,
                            best_val_loss=best_val_loss
                        )
                        logger.info(f"New best model saved (val_loss={val_loss:.4f})")
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= self.training_config.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {patience_counter} "
                            f"validations without improvement"
                        )
                        return params, self.normalizer
                
                # Periodic checkpoint with full state
                if step % self.training_config.checkpoint_frequency == 0:
                    self._save_checkpoint(
                        params, f"checkpoint_step_{step}.pkl",
                        opt_state=opt_state,
                        epoch=epoch,
                        step=step,
                        best_val_loss=best_val_loss
                    )
            
            # Stop prefetcher if used
            if use_prefetch and isinstance(data_iter, DataPrefetcher):
                data_iter.stop()
            
            # Log epoch summary
            avg_epoch_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} completed: avg_loss={avg_epoch_loss:.4f}")
        
        logger.info("Training completed!")
        return params, self.normalizer
    
    def _validate(
        self,
        params: hk.Params,
        val_data: xr.Dataset,
        rng: jax.random.PRNGKey,
    ) -> float:
        """Evaluate model on validation set (optimized with JIT).
        
        Args:
            params: Model parameters.
            val_data: Validation dataset (normalized).
            rng: Random key.
            
        Returns:
            Average validation loss.
        """
        # Define downstream region for target cropping
        downstream_region = (
            self.region_config.downstream_lat_min,
            self.region_config.downstream_lat_max,
            self.region_config.downstream_lon_min,
            self.region_config.downstream_lon_max,
        )
        
        val_losses = []
        
        for input_data, target_data in create_sliding_windows(val_data, downstream_region=downstream_region):
            rng, step_rng = jax.random.split(rng)
            
            # Evaluation step
            # Note: The forward pass is JIT compiled internally via Haiku
            loss = self._eval_step(params, input_data, target_data, step_rng)
            val_losses.append(float(jax.device_get(loss)))
        
        return np.mean(val_losses) if val_losses else float('inf')
    
    def _save_checkpoint(
        self, 
        params: hk.Params, 
        filename: str,
        opt_state: Optional[optax.OptState] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        best_val_loss: Optional[float] = None,
    ) -> None:
        """Save model checkpoint with training state.
        
        Args:
            params: Model parameters to save.
            filename: Checkpoint filename.
            opt_state: Optimizer state (optional, for resuming).
            epoch: Current epoch number (optional).
            step: Current step number (optional).
            best_val_loss: Best validation loss so far (optional).
        """
        checkpoint_path = os.path.join(self.output_dir, filename)
        
        checkpoint_data = {
            'params': params,
            'opt_state': opt_state,
            'epoch': epoch,
            'step': step,
            'best_val_loss': best_val_loss,
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")



def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load model checkpoint from file.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        Dictionary containing checkpoint data (params, opt_state, epoch, step, etc.).
        For backward compatibility, if checkpoint only contains params, returns {'params': params}.
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Handle old checkpoint format (just params, not wrapped in dict with 'params' key)
    if isinstance(checkpoint_data, dict) and 'params' not in checkpoint_data:
        # Old format: dict is the params itself (Haiku params are dicts)
        logger.warning("Loading old checkpoint format (params dict without 'params' key)")
        checkpoint_data = {'params': checkpoint_data}
    elif not isinstance(checkpoint_data, dict):
        # Very old format: params in some other structure
        logger.warning("Loading old checkpoint format (non-dict params)")
        checkpoint_data = {'params': checkpoint_data}
    # else: New format with 'params' key, use as-is
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint_data


def load_normalizer(normalizer_path: str) -> DataNormalizer:
    """Load normalizer from file.
    
    Args:
        normalizer_path: Path to normalizer file.
        
    Returns:
        DataNormalizer instance.
    """
    normalizer = DataNormalizer()
    normalizer.load(normalizer_path)
    return normalizer
