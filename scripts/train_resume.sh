#!/bin/bash
# Resume训练脚本 - 从checkpoint恢复训练
# Resume training script - continue from checkpoint

set -e

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法 / Usage: $0 <checkpoint_path> [output_dir]"
    echo ""
    echo "示例 / Example:"
    echo "  $0 checkpoints/experiment_1/checkpoint_step_5000.pkl"
    echo "  $0 checkpoints/experiment_1/best_model.pkl checkpoints/experiment_1_continued"
    echo ""
    exit 1
fi

CHECKPOINT_PATH=$1
OUTPUT_DIR=${2:-"checkpoints/resumed_$(date +%Y%m%d_%H%M%S)"}

# 检查checkpoint文件
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint文件不存在 - $CHECKPOINT_PATH"
    exit 1
fi

echo "========================================================================"
echo "Resume训练模式 / Resume Training Mode"
echo "========================================================================"
echo ""
echo "从checkpoint恢复训练 / Resuming from checkpoint:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo ""

python scripts/train_model.py \
    --data data/processed/regional_weather.nc \
    --output-dir "$OUTPUT_DIR" \
    --resume-from "$CHECKPOINT_PATH" \
    --num-epochs 100 \
    --use-prefetch \
    --prefetch-buffer-size 8

echo ""
echo "✓ Resume训练完成 / Resume training completed"
