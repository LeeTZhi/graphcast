#!/bin/bash
# 修复过拟合问题的训练脚本

python3 train_convlstm.py \
  --data ../Try/2019-2023_clean_mask.nc \
  --output ./checkpoint/convlstm_attention_deep_fixed \
  --device cuda \
  --train-ratio 0.85 \
  --test-start-date 2024-01-01 \
  --early-stopping-patience 50 \
  --num-epochs 400 \
  --high-precip-threshold 10 \
  --high-precip-weight 3.0 \
  --dropout-rate 0.2 \
  --learning-rate 0.0001 \
  --batch-size 16 \
  --weight-decay 1e-4 \
  --gradient-clip-norm 1.0 \
  --use-amp \
  --model-type deep \
  --no-batch-norm \
  --use-spatial-dropout \
  --use-group-norm \
  --use-attention

# 关键修改：
# 1. learning-rate: 0.003 -> 0.0001 (降低30倍，避免过拟合)
# 2. dropout-rate: 0.01 -> 0.2 (增强正则化)
# 3. batch-size: 32 -> 16 (增加训练噪声)
# 4. high-precip-weight: 5 -> 3 (降低极端事件权重)
# 5. weight-decay: 默认1e-5 -> 1e-4 (增强L2正则化)
# 6. early-stopping-patience: 400 -> 50 (更早停止)
# 7. 启用 spatial-dropout (更适合空间数据)
