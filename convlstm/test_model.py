"""Tests for ConvLSTM model components.

This module contains unit tests for ConvLSTMCell and ConvLSTMUNet.
"""

import torch
import pytest
from convlstm.model import ConvLSTMCell, ConvLSTMUNet


class TestConvLSTMCell:
    """Unit tests for ConvLSTMCell."""
    
    def test_cell_initialization(self):
        """Test ConvLSTMCell initializes correctly."""
        cell = ConvLSTMCell(input_dim=56, hidden_dim=32, kernel_size=3)
        assert cell.input_dim == 56
        assert cell.hidden_dim == 32
        assert cell.kernel_size == 3
        assert cell.padding == 1
    
    def test_cell_forward_shape(self):
        """Test ConvLSTMCell forward pass produces correct output shape."""
        batch_size = 2
        input_dim = 56
        hidden_dim = 32
        height = 20
        width = 20
        
        cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=3)
        
        # Create input and initial states
        input_tensor = torch.randn(batch_size, input_dim, height, width)
        h_cur = torch.zeros(batch_size, hidden_dim, height, width)
        c_cur = torch.zeros(batch_size, hidden_dim, height, width)
        
        # Forward pass
        h_next, c_next = cell(input_tensor, (h_cur, c_cur))
        
        # Check output shapes
        assert h_next.shape == (batch_size, hidden_dim, height, width)
        assert c_next.shape == (batch_size, hidden_dim, height, width)
    
    def test_cell_init_hidden(self):
        """Test ConvLSTMCell hidden state initialization."""
        batch_size = 4
        hidden_dim = 32
        height = 20
        width = 20
        device = torch.device('cpu')
        
        cell = ConvLSTMCell(input_dim=56, hidden_dim=hidden_dim, kernel_size=3)
        h, c = cell.init_hidden(batch_size, height, width, device)
        
        assert h.shape == (batch_size, hidden_dim, height, width)
        assert c.shape == (batch_size, hidden_dim, height, width)
        assert torch.all(h == 0)
        assert torch.all(c == 0)


class TestConvLSTMUNet:
    """Unit tests for ConvLSTMUNet."""
    
    def test_model_initialization(self):
        """Test ConvLSTMUNet initializes correctly."""
        model = ConvLSTMUNet(
            input_channels=56,
            hidden_channels=[32, 64],
            output_channels=1,
            kernel_size=3
        )
        assert model.input_channels == 56
        assert model.hidden_channels == [32, 64]
        assert model.output_channels == 1
        assert model.kernel_size == 3
    
    def test_model_initialization_default_params(self):
        """Test ConvLSTMUNet initializes with default parameters."""
        model = ConvLSTMUNet()
        assert model.input_channels == 56
        assert model.hidden_channels == [32, 64]
        assert model.output_channels == 1
        assert model.kernel_size == 3
    
    def test_model_initialization_invalid_hidden_channels(self):
        """Test ConvLSTMUNet raises error with invalid hidden_channels."""
        with pytest.raises(ValueError, match="hidden_channels must contain at least 2 values"):
            ConvLSTMUNet(hidden_channels=[32])
    
    def test_model_forward_shape(self):
        """Test ConvLSTMUNet forward pass produces correct output shape."""
        batch_size = 2
        time_steps = 6
        channels = 56
        height = 20
        width = 20
        
        model = ConvLSTMUNet(
            input_channels=channels,
            hidden_channels=[32, 64],
            output_channels=1
        )
        
        # Create input tensor
        x = torch.randn(batch_size, time_steps, channels, height, width)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1, height, width)
    
    def test_model_forward_non_negative_output(self):
        """Test ConvLSTMUNet produces non-negative outputs (ReLU activation)."""
        batch_size = 2
        time_steps = 6
        channels = 56
        height = 20
        width = 20
        
        model = ConvLSTMUNet(
            input_channels=channels,
            hidden_channels=[32, 64],
            output_channels=1
        )
        
        # Create input tensor
        x = torch.randn(batch_size, time_steps, channels, height, width)
        
        # Forward pass
        output = model(x)
        
        # Check all outputs are non-negative
        assert torch.all(output >= 0)
    
    def test_model_forward_different_spatial_sizes(self):
        """Test ConvLSTMUNet works with different spatial dimensions."""
        batch_size = 2
        time_steps = 6
        channels = 56
        
        # Test with different spatial sizes
        for height, width in [(16, 16), (24, 32), (40, 40)]:
            model = ConvLSTMUNet(
                input_channels=channels,
                hidden_channels=[32, 64],
                output_channels=1
            )
            
            x = torch.randn(batch_size, time_steps, channels, height, width)
            output = model(x)
            
            assert output.shape == (batch_size, 1, height, width)
    
    def test_model_forward_different_hidden_channels(self):
        """Test ConvLSTMUNet works with different hidden channel configurations."""
        batch_size = 2
        time_steps = 6
        channels = 56
        height = 20
        width = 20
        
        # Test with different hidden channel configurations
        for hidden_channels in [[16, 32], [32, 64], [64, 128]]:
            model = ConvLSTMUNet(
                input_channels=channels,
                hidden_channels=hidden_channels,
                output_channels=1
            )
            
            x = torch.randn(batch_size, time_steps, channels, height, width)
            output = model(x)
            
            assert output.shape == (batch_size, 1, height, width)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
