#!/bin/bash
# Regional Weather Prediction - Training Script
# 区域天气预测 - 训练脚本

set -e  # Exit on error

# ============================================================================
# 配置参数 / Configuration
# ============================================================================

# 数据路径 / Data paths
DATA_PATH="data/processed/regional_weather.nc"
OUTPUT_DIR="checkpoints/experiment_$(date +%Y%m%d_%H%M%S)"

# 区域配置 / Region configuration
DOWNSTREAM_LAT_MIN=25.0
DOWNSTREAM_LAT_MAX=40.0
DOWNSTREAM_LON_MIN=110.0
DOWNSTREAM_LON_MAX=125.0

UPSTREAM_LAT_MIN=25.0
UPSTREAM_LAT_MAX=50.0
UPSTREAM_LON_MIN=70.0
UPSTREAM_LON_MAX=110.0

# 模型架构 / Model architecture
LATENT_SIZE=256
NUM_GNN_LAYERS=12
MLP_HIDDEN_SIZE=256
MLP_NUM_HIDDEN_LAYERS=2

# 训练超参数 / Training hyperparameters
LEARNING_RATE=1e-4
BATCH_SIZE=8
NUM_EPOCHS=100
GRADIENT_CLIP_NORM=1.0
WEIGHT_DECAY=1e-5
WARMUP_STEPS=1000

# 损失函数 / Loss function
HIGH_PRECIP_THRESHOLD=10.0
HIGH_PRECIP_WEIGHT=3.0

# 训练循环 / Training loop
VALIDATION_FREQUENCY=500
CHECKPOINT_FREQUENCY=1000
EARLY_STOPPING_PATIENCE=10

# 数据划分 / Data split
TEST_START_DATE="2020-06-01"
TRAIN_RATIO=0.7
VAL_RATIO=0.3

# GPU优化 / GPU optimization
USE_PREFETCH=true
PREFETCH_BUFFER_SIZE=8

# 其他 / Other
SEED=42
VERBOSE=false

# Resume训练 / Resume training (留空表示从头开始)
RESUME_FROM=""

# ============================================================================
# 函数定义 / Function definitions
# ============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "错误: 文件不存在 - $1"
        echo "Error: File not found - $1"
        exit 1
    fi
}

check_gpu() {
    echo "检查GPU状态 / Checking GPU status..."
    python scripts/check_gpu.py
    if [ $? -ne 0 ]; then
        echo "警告: GPU未检测到，将使用CPU训练（速度较慢）"
        echo "Warning: GPU not detected, will use CPU (slower)"
        read -p "是否继续? Continue? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# ============================================================================
# 主程序 / Main program
# ============================================================================

print_header "区域天气预测模型训练 / Regional Weather Prediction Training"

# 检查数据文件
echo "检查数据文件 / Checking data file..."
check_file "$DATA_PATH"
echo "✓ 数据文件存在 / Data file exists"

# 检查GPU
check_gpu

# 创建输出目录
echo ""
echo "创建输出目录 / Creating output directory..."
mkdir -p "$OUTPUT_DIR"
echo "✓ 输出目录: $OUTPUT_DIR"

# 保存配置
CONFIG_FILE="$OUTPUT_DIR/training_config.txt"
echo "保存训练配置 / Saving training configuration..."
cat > "$CONFIG_FILE" << EOF
Training Configuration
======================
Date: $(date)
Host: $(hostname)

Data:
  - Data path: $DATA_PATH
  - Output directory: $OUTPUT_DIR

Region:
  - Downstream: lat[$DOWNSTREAM_LAT_MIN, $DOWNSTREAM_LAT_MAX], lon[$DOWNSTREAM_LON_MIN, $DOWNSTREAM_LON_MAX]
  - Upstream: lat[$UPSTREAM_LAT_MIN, $UPSTREAM_LAT_MAX], lon[$UPSTREAM_LON_MIN, $UPSTREAM_LON_MAX]

Model:
  - Latent size: $LATENT_SIZE
  - GNN layers: $NUM_GNN_LAYERS
  - MLP hidden size: $MLP_HIDDEN_SIZE
  - MLP hidden layers: $MLP_NUM_HIDDEN_LAYERS

Training:
  - Learning rate: $LEARNING_RATE
  - Batch size: $BATCH_SIZE
  - Epochs: $NUM_EPOCHS
  - Gradient clip: $GRADIENT_CLIP_NORM
  - Weight decay: $WEIGHT_DECAY
  - Warmup steps: $WARMUP_STEPS

Loss:
  - High precip threshold: $HIGH_PRECIP_THRESHOLD mm
  - High precip weight: $HIGH_PRECIP_WEIGHT

Data Split:
  - Test start date: $TEST_START_DATE
  - Train ratio: $TRAIN_RATIO
  - Val ratio: $VAL_RATIO

GPU Optimization:
  - Use prefetch: $USE_PREFETCH
  - Prefetch buffer: $PREFETCH_BUFFER_SIZE

Other:
  - Seed: $SEED
  - Resume from: ${RESUME_FROM:-"None (training from scratch)"}
EOF
echo "✓ 配置已保存到: $CONFIG_FILE"

# 构建训练命令
print_header "开始训练 / Starting Training"

TRAIN_CMD="python scripts/train_model.py \
    --data $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --downstream-lat-min $DOWNSTREAM_LAT_MIN \
    --downstream-lat-max $DOWNSTREAM_LAT_MAX \
    --downstream-lon-min $DOWNSTREAM_LON_MIN \
    --downstream-lon-max $DOWNSTREAM_LON_MAX \
    --upstream-lat-min $UPSTREAM_LAT_MIN \
    --upstream-lat-max $UPSTREAM_LAT_MAX \
    --upstream-lon-min $UPSTREAM_LON_MIN \
    --upstream-lon-max $UPSTREAM_LON_MAX \
    --latent-size $LATENT_SIZE \
    --num-gnn-layers $NUM_GNN_LAYERS \
    --mlp-hidden-size $MLP_HIDDEN_SIZE \
    --mlp-num-hidden-layers $MLP_NUM_HIDDEN_LAYERS \
    --learning-rate $LEARNING_RATE \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --gradient-clip-norm $GRADIENT_CLIP_NORM \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --high-precip-threshold $HIGH_PRECIP_THRESHOLD \
    --high-precip-weight $HIGH_PRECIP_WEIGHT \
    --validation-frequency $VALIDATION_FREQUENCY \
    --checkpoint-frequency $CHECKPOINT_FREQUENCY \
    --early-stopping-patience $EARLY_STOPPING_PATIENCE \
    --test-start-date $TEST_START_DATE \
    --train-ratio $TRAIN_RATIO \
    --val-ratio $VAL_RATIO \
    --prefetch-buffer-size $PREFETCH_BUFFER_SIZE \
    --seed $SEED"

# 添加可选参数
if [ "$USE_PREFETCH" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use-prefetch"
else
    TRAIN_CMD="$TRAIN_CMD --no-prefetch"
fi

if [ "$VERBOSE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --verbose"
fi

if [ -n "$RESUME_FROM" ]; then
    check_file "$RESUME_FROM"
    TRAIN_CMD="$TRAIN_CMD --resume-from $RESUME_FROM"
fi

# 显示命令
echo "训练命令 / Training command:"
echo "$TRAIN_CMD"
echo ""

# 执行训练
START_TIME=$(date +%s)
eval $TRAIN_CMD
EXIT_CODE=$?
END_TIME=$(date +%s)

# 计算训练时间
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

# 训练结果
echo ""
print_header "训练完成 / Training Completed"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练成功完成 / Training completed successfully"
    echo "✓ 训练时间 / Training time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "输出文件 / Output files:"
    echo "  - 最佳模型 / Best model: $OUTPUT_DIR/best_model.pkl"
    echo "  - 归一化器 / Normalizer: $OUTPUT_DIR/normalizer.pkl"
    echo "  - 训练日志 / Training log: $OUTPUT_DIR/training.log"
    echo "  - 配置文件 / Config file: $CONFIG_FILE"
    echo ""
    echo "下一步 / Next steps:"
    echo "  1. 查看训练日志: cat $OUTPUT_DIR/training.log"
    echo "  2. 运行推理: python scripts/run_inference.py --checkpoint $OUTPUT_DIR/best_model.pkl ..."
    echo "  3. 评估模型: python scripts/evaluate_model.py --checkpoint $OUTPUT_DIR/best_model.pkl ..."
else
    echo "✗ 训练失败 / Training failed with exit code: $EXIT_CODE"
    echo "✗ 请检查日志: $OUTPUT_DIR/training.log"
    exit $EXIT_CODE
fi
