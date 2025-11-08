#!/bin/bash
# 快速训练脚本 - 用于测试和调试
# Quick training script - for testing and debugging

set -e

echo "========================================================================"
echo "快速训练模式 / Quick Training Mode"
echo "========================================================================"
echo ""
echo "此脚本使用较小的模型和较少的epoch，适合快速测试"
echo "This script uses smaller model and fewer epochs for quick testing"
echo ""

python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir checkpoints/quick_test_$(date +%Y%m%d_%H%M%S) \
    --latent-size 128 \
    --num-gnn-layers 6 \
    --batch-size 4 \
    --num-epochs 10 \
    --validation-frequency 100 \
    --checkpoint-frequency 500 \
    --use-prefetch \
    --prefetch-buffer-size 4 \
    --verbose

echo ""
echo "✓ 快速训练完成 / Quick training completed"
