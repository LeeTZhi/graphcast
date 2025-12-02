"""Rolling forecast functionality for multi-step weather prediction.

This module implements autoregressive rolling forecasts where model predictions
are fed back as inputs for subsequent timesteps, enabling multi-step predictions
up to 6 timesteps ahead.
"""

import logging
from typing import Union, Dict

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def rolling_forecast(
    model: nn.Module,
    initial_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    num_steps: int,
    device: torch.device
) -> torch.Tensor:
    """Perform rolling forecast by feeding predictions back as inputs.
    
    This function implements autoregressive prediction where the model's output
    at timestep t becomes part of the input for predicting timestep t+1. The
    input window is shifted by removing the oldest timestep and appending the
    latest prediction.
    
    Args:
        model: Trained ConvLSTM model in eval mode. Must be configured for
               multi-variable mode (56 output channels).
        initial_input: Initial input sequence. Can be:
                      - Tensor [B, T, C, H, W] for single-stream models
                      - Dict with 'downstream' and 'upstream' keys for dual-stream
                      where B=batch size, T=window size (typically 6),
                      C=channels (56), H,W=spatial dimensions
        num_steps: Number of future steps to predict (1-6).
        device: Device to run inference on (e.g., torch.device('cuda')).
    
    Returns:
        predictions: Tensor [B, num_steps, C_out, H, W] where C_out is the
                    number of output channels (1 for single-variable, 56 for
                    multi-variable mode).
    
    Raises:
        NotImplementedError: If model is in single-variable mode (1 output channel)
                           or if dual-stream mode is used.
        ValueError: If num_steps is not between 1 and 6.
    
    Example:
        >>> # Load model in multi-variable mode
        >>> model = ConvLSTMUNet(input_channels=56, hidden_channels=[64, 128],
        ...                      output_channels=56, kernel_size=3, multi_variable=True)
        >>> model.eval()
        >>> 
        >>> # Prepare initial input [1, 6, 56, 64, 64]
        >>> initial_input = torch.randn(1, 6, 56, 64, 64)
        >>> 
        >>> # Generate 3-step rolling forecast
        >>> predictions = rolling_forecast(
        ...     model=model,
        ...     initial_input=initial_input,
        ...     num_steps=3,
        ...     device=torch.device('cuda')
        ... )
        >>> print(predictions.shape)  # [1, 3, 56, 64, 64]
    
    Note:
        - Only supports multi-variable mode where output channels match input channels (56)
        - Single-variable mode is not supported because predictions (1 channel) cannot
          be fed back as inputs (56 channels required)
        - Dual-stream mode is not yet supported as it requires upstream data for
          future timesteps
        - The model must be in evaluation mode before calling this function
    """
    # Validate num_steps
    if not (1 <= num_steps <= 6):
        raise ValueError(
            f"num_steps must be between 1 and 6, got {num_steps}"
        )
    
    # Check if input is dual-stream (dict)
    if isinstance(initial_input, dict):
        raise NotImplementedError(
            "Rolling forecast is not yet supported for dual-stream mode. "
            "Dual-stream models require upstream data for future timesteps, "
            "which is not available during rolling prediction."
        )
    
    # Move input to device
    current_input = initial_input.to(device)
    
    # Get input dimensions
    batch_size, window_size, input_channels, height, width = current_input.size()
    
    # Validate input channels
    if input_channels != 56:
        raise ValueError(
            f"Expected 56 input channels for multi-variable mode, got {input_channels}"
        )
    
    # Set model to eval mode (defensive)
    model.eval()
    
    # Perform a test forward pass to check output channels
    with torch.no_grad():
        test_output = model(current_input)  # [B, C_out, H, W]
        output_channels = test_output.size(1)
    
    # Check if model is in single-variable mode
    if output_channels == 1:
        raise NotImplementedError(
            "Rolling forecast is not supported for single-variable mode. "
            "Single-variable models output 1 channel (precipitation only), "
            "but rolling forecast requires 56 channels to feed back as input. "
            "Please use a model configured with multi_variable=True."
        )
    
    # Validate output channels match input channels for autoregressive prediction
    if output_channels != input_channels:
        raise ValueError(
            f"Output channels ({output_channels}) must match input channels "
            f"({input_channels}) for rolling forecast. The model must be in "
            f"multi-variable mode with 56 output channels."
        )
    
    logger.info(f"Starting rolling forecast for {num_steps} steps")
    logger.info(f"Input shape: {current_input.shape}")
    logger.info(f"Output channels: {output_channels}")
    
    # Collect predictions
    predictions = []
    
    # Perform rolling prediction
    with torch.no_grad():
        for step in range(num_steps):
            # Predict next timestep
            pred = model(current_input)  # [B, C_out, H, W]
            predictions.append(pred)
            
            logger.debug(f"Step {step + 1}/{num_steps}: prediction shape {pred.shape}")
            
            # Update input window for next prediction
            # Shift window: remove oldest timestep, append prediction
            # pred is [B, C_out, H, W], need to expand to [B, 1, C_out, H, W]
            pred_expanded = pred.unsqueeze(1)  # [B, 1, C_out, H, W]
            
            # Remove first timestep and append prediction
            # current_input[:, 1:, :, :, :] removes the oldest timestep [B, T-1, C, H, W]
            # Concatenate with new prediction [B, 1, C_out, H, W]
            current_input = torch.cat([
                current_input[:, 1:, :, :, :],  # [B, T-1, C, H, W]
                pred_expanded  # [B, 1, C_out, H, W]
            ], dim=1)  # [B, T, C, H, W]
            
            # Verify window size is maintained
            assert current_input.size(1) == window_size, \
                f"Window size changed from {window_size} to {current_input.size(1)}"
    
    # Stack predictions along time dimension
    predictions = torch.stack(predictions, dim=1)  # [B, num_steps, C_out, H, W]
    
    logger.info(f"Rolling forecast complete. Output shape: {predictions.shape}")
    
    return predictions
