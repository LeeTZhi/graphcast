#!/bin/bash
# GPU优化训练脚本 - 最大化GPU利用率
# GPU optimized training script - maximize GPU utilization

set -e

echo "========================================================================"
echo "GPU优化训练模式 / GPU Optimized Training Mode"
echo "========================================================================"
echo ""

# 检查GPU
echo "检查GPU状态..."
python scripts/check_gpu.py || {
    echo "错误: 未检测到GPU"
    exit 1
}

echo ""
echo "使用GPU优化配置进行训练..."
echo "Training with GPU optimized configuration..."
echo ""

python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir checkpoints/gpu_optimized_$(date +%Y%m%d_%H%M%S) \
    --latent-size 256 \
    --num-gnn-layers 12 \
    --batch-size 16 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --use-prefetch \
    --prefetch-buffer-size 16 \
    --validation-frequency 500 \
    --checkpoint-frequency 1000 \
    --jax-platform gpu

echo ""
echo "✓ GPU优化训练完成 / GPU optimized training completed"
