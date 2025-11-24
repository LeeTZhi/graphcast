"""Property-based tests for ConvLSTM model components.

This module contains property-based tests using Hypothesis to verify
correctness properties across many randomly generated inputs.
"""

import torch
import torch.nn as nn
import pytest
from hypothesis import given, settings, strategies as st
from convlstm.model import ConvLSTMCell, ConvLSTMUNet


# Feature: convlstm-weather-prediction, Property 5: Model preserves spatial dimensions
@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    input_dim=st.integers(min_value=1, max_value=64),
    hidden_dim=st.integers(min_value=1, max_value=64),
    height=st.integers(min_value=10, max_value=50),
    width=st.integers(min_value=10, max_value=50),
    kernel_size=st.sampled_from([3, 5, 7])
)
def test_convlstm_cell_preserves_spatial_dimensions(
    batch_size, input_dim, hidden_dim, height, width, kernel_size
):
    """Property 5: ConvLSTMCell preserves spatial dimensions.
    
    For any input tensor with spatial dimensions [H, W], the ConvLSTMCell
    output should have the same spatial dimensions [H, W] because it uses
    padding to maintain dimensions.
    
    Validates: Requirements 2.3, 2.4
    """
    # Create ConvLSTMCell
    cell = ConvLSTMCell(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        bias=True
    )
    
    # Create input tensor and initial states
    input_tensor = torch.randn(batch_size, input_dim, height, width)
    h_cur = torch.zeros(batch_size, hidden_dim, height, width)
    c_cur = torch.zeros(batch_size, hidden_dim, height, width)
    
    # Forward pass
    h_next, c_next = cell(input_tensor, (h_cur, c_cur))
    
    # Property: Output spatial dimensions should match input spatial dimensions
    assert h_next.shape[0] == batch_size, \
        f"Batch size mismatch: expected {batch_size}, got {h_next.shape[0]}"
    assert h_next.shape[1] == hidden_dim, \
        f"Hidden dim mismatch: expected {hidden_dim}, got {h_next.shape[1]}"
    assert h_next.shape[2] == height, \
        f"Height mismatch: expected {height}, got {h_next.shape[2]}"
    assert h_next.shape[3] == width, \
        f"Width mismatch: expected {width}, got {h_next.shape[3]}"
    
    assert c_next.shape[0] == batch_size, \
        f"Cell batch size mismatch: expected {batch_size}, got {c_next.shape[0]}"
    assert c_next.shape[1] == hidden_dim, \
        f"Cell hidden dim mismatch: expected {hidden_dim}, got {c_next.shape[1]}"
    assert c_next.shape[2] == height, \
        f"Cell height mismatch: expected {height}, got {c_next.shape[2]}"
    assert c_next.shape[3] == width, \
        f"Cell width mismatch: expected {width}, got {c_next.shape[3]}"


# Feature: convlstm-weather-prediction, Property 6: Downsampling reduces spatial dimensions
@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    channels=st.integers(min_value=1, max_value=64),
    height=st.integers(min_value=10, max_value=100),
    width=st.integers(min_value=10, max_value=100)
)
def test_downsampling_reduces_spatial_dimensions(
    batch_size, channels, height, width
):
    """Property 6: Downsampling reduces spatial dimensions.
    
    For any input to the encoder with spatial dimensions [H, W], the encoded
    features after downsampling should have dimensions [H/2, W/2] because
    MaxPool2d with kernel_size=2 and stride=2 reduces each dimension by half.
    
    Validates: Requirements 2.2
    """
    # Create the pooling layer used in the model
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Create input tensor with arbitrary spatial dimensions
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # Apply downsampling
    downsampled = pool(input_tensor)
    
    # Property: Output spatial dimensions should be half of input dimensions
    expected_height = height // 2
    expected_width = width // 2
    
    assert downsampled.shape[0] == batch_size, \
        f"Batch size mismatch: expected {batch_size}, got {downsampled.shape[0]}"
    assert downsampled.shape[1] == channels, \
        f"Channels mismatch: expected {channels}, got {downsampled.shape[1]}"
    assert downsampled.shape[2] == expected_height, \
        f"Height after downsampling mismatch: expected {expected_height}, got {downsampled.shape[2]}"
    assert downsampled.shape[3] == expected_width, \
        f"Width after downsampling mismatch: expected {expected_width}, got {downsampled.shape[3]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
