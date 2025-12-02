"""Simple test script for ConvLSTM model components.

This script tests the ConvLSTMCell and ConvLSTMUNet without requiring pytest.
Run with: python convlstm/test_model_simple.py
"""

import sys
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
except ImportError:
    print("✗ PyTorch not installed. Please install PyTorch to use ConvLSTM models.")
    print("  Install with: pip install torch")
    sys.exit(1)

from convlstm.model import ConvLSTMCell, ConvLSTMUNet


def test_convlstm_cell():
    """Test ConvLSTMCell functionality."""
    print("\n=== Testing ConvLSTMCell ===")
    
    # Test 1: Initialization
    print("Test 1: Cell initialization...")
    cell = ConvLSTMCell(input_dim=56, hidden_dim=32, kernel_size=3)
    assert cell.input_dim == 56
    assert cell.hidden_dim == 32
    assert cell.kernel_size == 3
    assert cell.padding == 1
    print("  ✓ Cell initialized correctly")
    
    # Test 2: Forward pass shape
    print("Test 2: Forward pass shape...")
    batch_size = 2
    input_dim = 56
    hidden_dim = 32
    height = 20
    width = 20
    
    input_tensor = torch.randn(batch_size, input_dim, height, width)
    h_cur = torch.zeros(batch_size, hidden_dim, height, width)
    c_cur = torch.zeros(batch_size, hidden_dim, height, width)
    
    h_next, c_next = cell(input_tensor, (h_cur, c_cur))
    
    assert h_next.shape == (batch_size, hidden_dim, height, width)
    assert c_next.shape == (batch_size, hidden_dim, height, width)
    print(f"  ✓ Output shapes correct: h={h_next.shape}, c={c_next.shape}")
    
    # Test 3: Hidden state initialization
    print("Test 3: Hidden state initialization...")
    device = torch.device('cpu')
    h, c = cell.init_hidden(batch_size, height, width, device)
    
    assert h.shape == (batch_size, hidden_dim, height, width)
    assert c.shape == (batch_size, hidden_dim, height, width)
    assert torch.all(h == 0)
    assert torch.all(c == 0)
    print(f"  ✓ Hidden states initialized correctly: h={h.shape}, c={c.shape}")
    
    print("✓ All ConvLSTMCell tests passed!")


def test_convlstm_unet():
    """Test ConvLSTMUNet functionality."""
    print("\n=== Testing ConvLSTMUNet ===")
    
    # Test 1: Initialization
    print("Test 1: Model initialization...")
    model = ConvLSTMUNet(
        input_channels=56,
        hidden_channels=[32, 64],
        output_channels=1,
        kernel_size=3
    )
    assert model.input_channels == 56
    assert model.hidden_channels == [32, 64]
    assert model.output_channels == 1
    print("  ✓ Model initialized correctly")
    
    # Test 2: Default parameters
    print("Test 2: Default parameters...")
    model_default = ConvLSTMUNet()
    assert model_default.input_channels == 56
    assert model_default.hidden_channels == [32, 64]
    print("  ✓ Default parameters work correctly")
    
    # Test 3: Invalid hidden channels
    print("Test 3: Invalid hidden channels...")
    try:
        ConvLSTMUNet(hidden_channels=[32])
        print("  ✗ Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 4: Forward pass shape
    print("Test 4: Forward pass shape...")
    batch_size = 2
    time_steps = 6
    channels = 56
    height = 20
    width = 20
    
    x = torch.randn(batch_size, time_steps, channels, height, width)
    output = model(x)
    
    assert output.shape == (batch_size, 1, height, width)
    print(f"  ✓ Output shape correct: {output.shape}")
    
    # Test 5: Output range check
    print("Test 5: Output range check...")
    assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf"
    assert torch.all(output >= -10) and torch.all(output <= 10), \
        f"Output out of reasonable range: min={output.min().item():.4f}, max={output.max().item():.4f}"
    print(f"  ✓ Outputs in reasonable range (min={output.min().item():.4f}, max={output.max().item():.4f})")
    
    # Test 6: Different spatial sizes
    print("Test 6: Different spatial sizes...")
    for height, width in [(16, 16), (24, 32), (40, 40)]:
        model_test = ConvLSTMUNet(input_channels=56, hidden_channels=[32, 64])
        x_test = torch.randn(2, 6, 56, height, width)
        output_test = model_test(x_test)
        assert output_test.shape == (2, 1, height, width)
    print(f"  ✓ Works with different spatial sizes")
    
    # Test 7: Different hidden channels
    print("Test 7: Different hidden channel configurations...")
    for hidden_channels in [[16, 32], [32, 64], [64, 128]]:
        model_test = ConvLSTMUNet(
            input_channels=56,
            hidden_channels=hidden_channels,
            output_channels=1
        )
        x_test = torch.randn(2, 6, 56, 20, 20)
        output_test = model_test(x_test)
        assert output_test.shape == (2, 1, 20, 20)
    print(f"  ✓ Works with different hidden channel configurations")
    
    print("✓ All ConvLSTMUNet tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("ConvLSTM Model Tests")
    print("=" * 60)
    
    try:
        test_convlstm_cell()
        test_convlstm_unet()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
