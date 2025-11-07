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
"""Inference pipeline for Regional Weather Prediction System."""

import logging
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr

from graphcast import typed_graph
from graphcast_regional.config import ModelConfig, RegionConfig
from graphcast_regional.model import RegionalGNN
from graphcast_regional.training import DataNormalizer


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Manages model inference workflow.
    
    Handles loading trained models, preparing input data, and generating
    predictions for single or multiple timesteps.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        region_config: RegionConfig,
        params: hk.Params,
        normalizer: DataNormalizer,
        graph: typed_graph.TypedGraph,
    ):
        """Initialize inference pipeline.
        
        Args:
            model_config: Model architecture configuration.
            region_config: Region boundaries configuration.
            params: Trained model parameters.
            normalizer: Fitted DataNormalizer for input preprocessing.
            graph: Regional graph structure.
        """
        self.model_config = model_config
        self.region_config = region_config
        self.params = params
        self.normalizer = normalizer
        self.graph = graph
        
        # Initialize model
        self._init_model()
        
        logger.info("Inference pipeline initialized")
    
    def _init_model(self):
        """Initialize Haiku model and transform."""
        def forward_fn(inputs: xr.Dataset, is_training: bool = False) -> xr.Dataset:
            model = RegionalGNN(self.model_config, self.region_config)
            return model(inputs, self.graph, is_training=is_training)
        
        self.forward = hk.transform(forward_fn)
    
    def predict(
        self,
        data: xr.Dataset,
        target_time: pd.Timestamp,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> xr.Dataset:
        """Generate single-step prediction for target time.
        
        Loads atmospheric data for time t-12h and t, applies normalization,
        and generates precipitation prediction for t+12h.
        
        Args:
            data: Full dataset containing historical atmospheric data.
            target_time: Target timestamp for prediction (t+12h).
            rng: Optional random key for stochastic models.
            
        Returns:
            xarray Dataset with precipitation predictions for downstream region.
            Dimensions: (lat, lon).
            
        Raises:
            ValueError: If required input timestamps are not available in data.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        logger.info(f"Generating prediction for target time: {target_time}")
        
        # Calculate required input timestamps: t-12h and t
        # Target time is t+12h, so t = target_time - 12h
        t = target_time - pd.Timedelta(hours=12)
        t_minus_12h = t - pd.Timedelta(hours=12)
        
        logger.info(f"Input timestamps: {t_minus_12h} and {t}")
        
        # Check if required timestamps exist in data
        available_times = pd.DatetimeIndex(data.time.values)
        
        if t_minus_12h not in available_times:
            raise ValueError(
                f"Required timestamp {t_minus_12h} not found in data. "
                f"Available times: {available_times[0]} to {available_times[-1]}"
            )
        
        if t not in available_times:
            raise ValueError(
                f"Required timestamp {t} not found in data. "
                f"Available times: {available_times[0]} to {available_times[-1]}"
            )
        
        # Extract input data for the two required timesteps
        input_data = data.sel(time=[t_minus_12h, t])
        
        # Apply normalization
        logger.info("Applying normalization to input data")
        input_data_norm = self.normalizer.normalize(input_data)
        
        # Generate prediction
        logger.info("Running model forward pass")
        prediction_norm = self.forward.apply(
            self.params, rng, input_data_norm, is_training=False
        )
        
        # Denormalize prediction
        logger.info("Denormalizing prediction")
        prediction = self.normalizer.denormalize(prediction_norm)
        
        # Clip negative precipitation values to zero
        prediction["precipitation"] = xr.where(
            prediction["precipitation"] < 0,
            0.0,
            prediction["precipitation"]
        )
        
        # Add target time as coordinate
        prediction = prediction.assign_coords({"time": target_time})
        
        logger.info(f"Prediction generated successfully for {target_time}")
        
        return prediction

    def predict_sequence(
        self,
        data: xr.Dataset,
        initial_time: pd.Timestamp,
        num_steps: int,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> xr.Dataset:
        """Generate multi-step predictions using autoregressive approach.
        
        Generates a sequence of predictions by using previous predictions
        as input for the next step. Each step predicts 12 hours ahead.
        
        Args:
            data: Full dataset containing historical atmospheric data.
            initial_time: Initial timestamp (t) for first prediction.
            num_steps: Number of 12-hour prediction steps to generate.
            rng: Optional random key for stochastic models.
            
        Returns:
            xarray Dataset with precipitation predictions for all timesteps.
            Dimensions: (time, lat, lon) where time has num_steps entries.
            
        Raises:
            ValueError: If required input timestamps are not available in data.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        logger.info(
            f"Generating {num_steps}-step prediction sequence starting from {initial_time}"
        )
        
        # Initialize list to store predictions
        predictions = []
        
        # Create a working copy of the data that we'll update with predictions
        working_data = data.copy(deep=True)
        
        # Current time starts at initial_time
        current_time = initial_time
        
        for step in range(num_steps):
            logger.info(f"Step {step + 1}/{num_steps}: predicting for {current_time + pd.Timedelta(hours=12)}")
            
            # Split random key for this step
            rng, step_rng = jax.random.split(rng)
            
            # Generate prediction for current_time + 12h
            target_time = current_time + pd.Timedelta(hours=12)
            
            try:
                prediction = self.predict(working_data, target_time, step_rng)
            except ValueError as e:
                logger.error(f"Failed to generate prediction at step {step + 1}: {e}")
                raise
            
            # Store prediction
            predictions.append(prediction)
            
            # Update working_data with the new prediction for use in next step
            # We need to add the predicted precipitation to the working dataset
            # at the target_time so it can be used as input for the next prediction
            
            # First, check if target_time already exists in working_data
            if target_time in pd.DatetimeIndex(working_data.time.values):
                # Update existing precipitation values for downstream region
                # Extract downstream region mask
                downstream_lat_mask = (
                    (working_data.lat >= self.region_config.downstream_lat_min) &
                    (working_data.lat <= self.region_config.downstream_lat_max)
                )
                downstream_lon_mask = (
                    (working_data.lon >= self.region_config.downstream_lon_min) &
                    (working_data.lon <= self.region_config.downstream_lon_max)
                )
                
                # Update precipitation at target_time for downstream region
                working_data["precipitation"].loc[
                    dict(
                        time=target_time,
                        lat=working_data.lat[downstream_lat_mask],
                        lon=working_data.lon[downstream_lon_mask]
                    )
                ] = prediction["precipitation"].values
            else:
                # If target_time doesn't exist, we need to create a new timestep
                # This is more complex as we need HPA variables too
                # For simplicity, we'll assume all required timesteps exist in data
                logger.warning(
                    f"Target time {target_time} not found in working data. "
                    f"Autoregressive prediction may be limited."
                )
            
            # Move to next timestep
            current_time = target_time
        
        # Concatenate all predictions along time dimension
        logger.info("Concatenating predictions into single dataset")
        predictions_combined = xr.concat(predictions, dim="time")
        
        logger.info(f"Multi-step prediction sequence completed: {num_steps} steps")
        
        return predictions_combined



def create_inference_pipeline(
    model_config: ModelConfig,
    region_config: RegionConfig,
    checkpoint_path: str,
    normalizer_path: str,
    graph: typed_graph.TypedGraph,
) -> InferencePipeline:
    """Create an inference pipeline from saved checkpoint and normalizer.
    
    Convenience function to load all required components and create
    an InferencePipeline instance.
    
    Args:
        model_config: Model architecture configuration.
        region_config: Region boundaries configuration.
        checkpoint_path: Path to saved model checkpoint (.pkl file).
        normalizer_path: Path to saved normalizer (.pkl file).
        graph: Regional graph structure.
        
    Returns:
        Initialized InferencePipeline ready for predictions.
    """
    import pickle
    
    logger.info("Creating inference pipeline from saved components")
    
    # Load model parameters
    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Handle both old (params only) and new (dict with params) checkpoint formats
    if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        # New format: dict with 'params' key
        params = checkpoint_data['params']
        if 'epoch' in checkpoint_data:
            logger.info(f"  Checkpoint from epoch {checkpoint_data['epoch']}, step {checkpoint_data.get('step', 'unknown')}")
        if 'best_val_loss' in checkpoint_data:
            logger.info(f"  Best validation loss: {checkpoint_data['best_val_loss']:.4f}")
    else:
        # Old format: just params (could be dict or other structure)
        params = checkpoint_data
        logger.info("  Using old checkpoint format (params only)")
    
    # Load normalizer
    logger.info(f"Loading normalizer from {normalizer_path}")
    normalizer = DataNormalizer()
    normalizer.load(normalizer_path)
    
    # Create inference pipeline
    pipeline = InferencePipeline(
        model_config=model_config,
        region_config=region_config,
        params=params,
        normalizer=normalizer,
        graph=graph,
    )
    
    logger.info("Inference pipeline created successfully")
    
    return pipeline


def batch_predict(
    pipeline: InferencePipeline,
    data: xr.Dataset,
    target_times: list[pd.Timestamp],
    rng: Optional[jax.random.PRNGKey] = None,
) -> xr.Dataset:
    """Generate predictions for multiple target times.
    
    Args:
        pipeline: Initialized InferencePipeline.
        data: Full dataset containing historical atmospheric data.
        target_times: List of target timestamps for predictions.
        rng: Optional random key for stochastic models.
        
    Returns:
        xarray Dataset with predictions for all target times.
        Dimensions: (time, lat, lon).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    logger.info(f"Generating batch predictions for {len(target_times)} timesteps")
    
    predictions = []
    
    for i, target_time in enumerate(target_times):
        logger.info(f"Processing {i + 1}/{len(target_times)}: {target_time}")
        
        # Split random key
        rng, step_rng = jax.random.split(rng)
        
        # Generate prediction
        try:
            prediction = pipeline.predict(data, target_time, step_rng)
            predictions.append(prediction)
        except ValueError as e:
            logger.warning(f"Skipping {target_time} due to error: {e}")
            continue
    
    if not predictions:
        raise ValueError("No valid predictions generated")
    
    # Concatenate predictions
    predictions_combined = xr.concat(predictions, dim="time")
    
    logger.info(f"Batch prediction completed: {len(predictions)} timesteps")
    
    return predictions_combined
