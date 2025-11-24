#!/usr/bin/env python3
"""
便捷训练脚本 - 提供多种训练模式
Convenient training script - provides multiple training modes
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示输出"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def train_standard(args):
    """标准训练模式"""
    output_dir = args.output_dir or f"checkpoints/standard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cmd = [
        "python", "scripts/train_model.py",
        "--data", args.data,
        "--output-dir", output_dir,
        "--latent-size", str(args.latent_size),
        "--num-gnn-layers", str(args.num_gnn_layers),
        "--batch-size", str(args.batch_size),
        "--num-epochs", str(args.num_epochs),
        "--learning-rate", str(args.learning_rate),
        "--use-prefetch",
        "--prefetch-buffer-size", str(args.prefetch_buffer),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    return run_command(cmd, "标准训练模式 / Standard Training Mode")


def train_quick(args):
    """快速训练模式 - 用于测试"""
    output_dir = f"checkpoints/quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cmd = [
        "python", "scripts/train_model.py",
        "--data", args.data,
        "--output-dir", output_dir,
        "--latent-size", "128",
        "--num-gnn-layers", "6",
        "--batch-size", "4",
        "--num-epochs", "10",
        "--validation-frequency", "100",
        "--checkpoint-frequency", "500",
        "--use-prefetch",
        "--verbose",
    ]
    
    return run_command(cmd, "快速训练模式 / Quick Training Mode")


def train_resume(args):
    """Resume训练模式"""
    if not args.checkpoint:
        print("错误: Resume模式需要指定 --checkpoint")
        return 1
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: Checkpoint文件不存在 - {checkpoint_path}")
        return 1
    
    output_dir = args.output_dir or f"checkpoints/resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cmd = [
        "python", "scripts/train_model.py",
        "--data", args.data,
        "--output-dir", output_dir,
        "--resume-from", str(checkpoint_path),
        "--num-epochs", str(args.num_epochs),
        "--use-prefetch",
        "--prefetch-buffer-size", str(args.prefetch_buffer),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    return run_command(cmd, "Resume训练模式 / Resume Training Mode")


def train_finetune(args):
    """Fine-tune训练模式"""
    if not args.checkpoint:
        print("错误: Fine-tune模式需要指定 --checkpoint")
        return 1
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: Checkpoint文件不存在 - {checkpoint_path}")
        return 1
    
    output_dir = args.output_dir or f"checkpoints/finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    learning_rate = args.learning_rate or 1e-5
    
    cmd = [
        "python", "scripts/train_model.py",
        "--data", args.data,
        "--output-dir", output_dir,
        "--resume-from", str(checkpoint_path),
        "--learning-rate", str(learning_rate),
        "--num-epochs", str(args.num_epochs or 50),
        "--early-stopping-patience", "5",
        "--use-prefetch",
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    return run_command(cmd, "Fine-tune训练模式 / Fine-tune Training Mode")


def train_gpu_optimized(args):
    """GPU优化训练模式"""
    # 检查GPU
    print("检查GPU状态...")
    result = subprocess.run(["python", "scripts/check_gpu.py"])
    if result.returncode != 0:
        print("\n警告: GPU检查失败，但将继续训练")
    
    output_dir = args.output_dir or f"checkpoints/gpu_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cmd = [
        "python", "scripts/train_model.py",
        "--data", args.data,
        "--output-dir", output_dir,
        "--latent-size", str(args.latent_size or 256),
        "--num-gnn-layers", str(args.num_gnn_layers or 12),
        "--batch-size", str(args.batch_size or 16),
        "--num-epochs", str(args.num_epochs),
        "--learning-rate", str(args.learning_rate),
        "--use-prefetch",
        "--prefetch-buffer-size", "16",
        "--jax-platform", "gpu",
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    return run_command(cmd, "GPU优化训练模式 / GPU Optimized Training Mode")


def main():
    parser = argparse.ArgumentParser(
        description="便捷训练脚本 / Convenient Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
训练模式 / Training Modes:
  standard      标准训练 / Standard training
  quick         快速测试训练 / Quick test training
  resume        从checkpoint恢复 / Resume from checkpoint
  finetune      Fine-tune训练 / Fine-tune training
  gpu           GPU优化训练 / GPU optimized training

示例 / Examples:
  # 标准训练
  python train.py standard --data data.nc --num-epochs 100

  # 快速测试
  python train.py quick --data data.nc

  # Resume训练
  python train.py resume --data data.nc --checkpoint checkpoints/exp1/best_model.pkl

  # Fine-tune
  python train.py finetune --data data.nc --checkpoint checkpoints/exp1/best_model.pkl --learning-rate 1e-5

  # GPU优化
  python train.py gpu --data data.nc --batch-size 16 --num-epochs 100
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["standard", "quick", "resume", "finetune", "gpu"],
        help="训练模式 / Training mode"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/regional_weather.nc",
        help="数据文件路径 / Data file path"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录 / Output directory (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint路径 (用于resume/finetune模式) / Checkpoint path (for resume/finetune mode)"
    )
    
    parser.add_argument(
        "--latent-size",
        type=int,
        default=256,
        help="潜在维度 / Latent size (default: 256)"
    )
    
    parser.add_argument(
        "--num-gnn-layers",
        type=int,
        default=12,
        help="GNN层数 / Number of GNN layers (default: 12)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小 / Batch size (default: 8)"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="训练轮数 / Number of epochs (default: 100)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="学习率 / Learning rate (default: 1e-4)"
    )
    
    parser.add_argument(
        "--prefetch-buffer",
        type=int,
        default=8,
        help="预取缓冲区大小 / Prefetch buffer size (default: 8)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出 / Verbose output"
    )
    
    args = parser.parse_args()
    
    # 检查数据文件
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 数据文件不存在 - {data_path}")
        return 1
    
    # 根据模式调用相应函数
    mode_functions = {
        "standard": train_standard,
        "quick": train_quick,
        "resume": train_resume,
        "finetune": train_finetune,
        "gpu": train_gpu_optimized,
    }
    
    return mode_functions[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
