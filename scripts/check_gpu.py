#!/usr/bin/env python3
"""Quick GPU check and configuration script."""

import sys

def check_gpu():
    """Check GPU availability and configuration."""
    print("=" * 80)
    print("GPU Configuration Check")
    print("=" * 80)
    print()
    
    # Check JAX
    try:
        import jax
        print(f"✓ JAX installed: version {jax.__version__}")
        print(f"  Backend: {jax.default_backend()}")
        print(f"  Devices: {jax.devices()}")
        
        if jax.default_backend() == 'gpu':
            print("  ✓ GPU backend is active")
            
            # Test GPU computation
            import jax.numpy as jnp
            x = jnp.ones((1000, 1000))
            y = jnp.dot(x, x)
            print(f"  ✓ GPU computation test passed")
        elif jax.default_backend() == 'cpu':
            print("  ⚠ WARNING: Running on CPU backend")
            print("  To use GPU, install JAX with CUDA support:")
            print("    pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        else:
            print(f"  Backend: {jax.default_backend()}")
            
    except ImportError:
        print("✗ JAX not installed")
        print("  Install with: pip install jax")
        return False
    
    print()
    
    # Check CUDA
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected:")
            # Parse nvidia-smi output for GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'Tesla' in line or 'GeForce' in line or 'RTX' in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ nvidia-smi command failed")
    except FileNotFoundError:
        print("✗ nvidia-smi not found (NVIDIA drivers not installed?)")
    
    print()
    
    # Check CUDA environment
    import os
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"✓ CUDA_HOME: {cuda_home}")
    else:
        print("⚠ CUDA_HOME not set")
    
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'cuda' in ld_library_path.lower():
        print(f"✓ LD_LIBRARY_PATH includes CUDA")
    else:
        print("⚠ LD_LIBRARY_PATH may not include CUDA libraries")
    
    print()
    
    # Check other dependencies
    try:
        import xarray
        print(f"✓ xarray: {xarray.__version__}")
    except ImportError:
        print("✗ xarray not installed")
    
    try:
        import haiku
        print(f"✓ haiku: {haiku.__version__}")
    except ImportError:
        print("✗ haiku not installed")
    
    try:
        import optax
        print(f"✓ optax: {optax.__version__}")
    except ImportError:
        print("✗ optax not installed")
    
    print()
    print("=" * 80)
    
    # Recommendations
    if jax.default_backend() == 'gpu':
        print("✓ System is ready for GPU training!")
        print()
        print("Recommended training command:")
        print("  python scripts/train_model.py \\")
        print("      --data data/processed/regional_weather.nc \\")
        print("      --output-dir checkpoints/gpu_run \\")
        print("      --batch-size 8 \\")
        print("      --use-prefetch \\")
        print("      --prefetch-buffer-size 8")
    else:
        print("⚠ GPU not available or not configured")
        print()
        print("To enable GPU:")
        print("1. Install NVIDIA drivers and CUDA toolkit")
        print("2. Install JAX with CUDA support:")
        print("   pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        print("3. Run this script again to verify")
    
    print("=" * 80)
    
    return jax.default_backend() == 'gpu'


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
