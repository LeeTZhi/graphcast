#!/bin/bash
# Fine-tune训练脚本 - 使用较小学习率继续训练
# Fine-tune training script - continue with smaller learning rate

set -e

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法 / Usage: $0 <checkpoint_path> [learning_rate]"
    echo ""
    echo "示例 / Example:"
    echo "  $0 checkpoints/experiment_1/best_model.pkl"
    echo "  $0 checkpoints/experiment_1/best_model.pkl 1e-5"
    echo ""
    exit 1
fi

CHECKPOINT_PATH=$1
LEARNING_RATE=${2:-1e-5}

# 检查checkpoint文件
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint文件不存在 - $CHECKPOINT_PATH"
    exit 1
fi

OUTPUT_DIR="checkpoints/finetuned_$(date +%Y%m%d_%H%M%S)"

echo "========================================================================"
echo "Fine-tune训练模式 / Fine-tune Training Mode"
echo "========================================================================"
echo ""
echo "使用较小学习率进行fine-tune / Fine-tuning with smaller learning rate:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Learning rate: $LEARNING_RATE"
echo "  Output dir: $OUTPUT_DIR"
echo ""

python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir "$OUTPUT_DIR" \
    --resume-from "$CHECKPOINT_PATH" \
    --learning-rate "$LEARNING_RATE" \
    --num-epochs 50 \
    --use-prefetch \
    --prefetch-buffer-size 8 \
    --early-stopping-patience 5

echo ""
echo "✓ Fine-tune训练完成 / Fine-tune training completed"
