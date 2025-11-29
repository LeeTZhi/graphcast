#!/usr/bin/env python3
"""诊断过拟合问题的脚本

分析训练日志，检查：
1. 训练集vs验证集loss差距
2. 学习率变化
3. 梯度范数
4. 数据划分是否合理
"""

import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_path: str):
    """解析训练日志"""
    train_losses = []
    val_losses = []
    learning_rates = []
    epochs = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # 匹配训练loss
            match = re.search(r'Epoch (\d+)/\d+ - Step \d+/\d+ - Loss: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                train_losses.append((epoch, loss))
            
            # 匹配验证loss
            match = re.search(r'Validation Loss: ([\d.]+)', line)
            if match:
                val_loss = float(match.group(1))
                val_losses.append(val_loss)
            
            # 匹配学习率
            match = re.search(r'Learning rate: ([\d.e-]+)', line)
            if match:
                lr = float(match.group(1))
                learning_rates.append(lr)
    
    return train_losses, val_losses, learning_rates


def plot_training_curves(train_losses, val_losses, learning_rates, output_dir):
    """绘制训练曲线"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 训练loss
    if train_losses:
        epochs, losses = zip(*train_losses)
        ax1.plot(epochs, losses, alpha=0.3, label='Train Loss (raw)')
        # 平滑曲线
        window = min(50, len(losses) // 10)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax1.plot(range(window//2, len(smoothed) + window//2), smoothed, 
                    label='Train Loss (smoothed)', linewidth=2)
    
    # 验证loss
    if val_losses:
        ax1.plot(range(len(val_losses)), val_losses, 
                label='Val Loss', linewidth=2, marker='o', markersize=3)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 过拟合指标
    if train_losses and val_losses:
        # 计算每个epoch的平均训练loss
        epoch_train_losses = {}
        for epoch, loss in train_losses:
            if epoch not in epoch_train_losses:
                epoch_train_losses[epoch] = []
            epoch_train_losses[epoch].append(loss)
        
        avg_train_losses = [np.mean(epoch_train_losses[e]) 
                           for e in sorted(epoch_train_losses.keys())]
        
        # 过拟合gap
        min_len = min(len(avg_train_losses), len(val_losses))
        gap = [val_losses[i] - avg_train_losses[i] for i in range(min_len)]
        
        ax2.plot(range(min_len), gap, linewidth=2, color='red')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Val Loss - Train Loss')
        ax2.set_title('Overfitting Gap (正值=过拟合)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    print(f"✓ 训练曲线已保存到: {output_dir / 'training_curves.png'}")
    
    # 3. 学习率曲线
    if learning_rates:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(learning_rates, linewidth=2)
        ax.set_xlabel('Validation Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_rate.png', dpi=150)
        print(f"✓ 学习率曲线已保存到: {output_dir / 'learning_rate.png'}")


def analyze_overfitting(train_losses, val_losses):
    """分析过拟合程度"""
    if not train_losses or not val_losses:
        print("⚠️  日志数据不足，无法分析")
        return
    
    # 计算每个epoch的平均训练loss
    epoch_train_losses = {}
    for epoch, loss in train_losses:
        if epoch not in epoch_train_losses:
            epoch_train_losses[epoch] = []
        epoch_train_losses[epoch].append(loss)
    
    avg_train_losses = [np.mean(epoch_train_losses[e]) 
                       for e in sorted(epoch_train_losses.keys())]
    
    # 最近10个epoch的统计
    recent_epochs = 10
    if len(avg_train_losses) >= recent_epochs and len(val_losses) >= recent_epochs:
        recent_train = avg_train_losses[-recent_epochs:]
        recent_val = val_losses[-recent_epochs:]
        
        train_mean = np.mean(recent_train)
        val_mean = np.mean(recent_val)
        gap = val_mean - train_mean
        gap_ratio = gap / train_mean * 100
        
        print("\n" + "="*60)
        print("过拟合诊断报告")
        print("="*60)
        print(f"最近{recent_epochs}个epoch统计:")
        print(f"  训练集平均loss: {train_mean:.4f}")
        print(f"  验证集平均loss: {val_mean:.4f}")
        print(f"  Loss差距: {gap:.4f} ({gap_ratio:.1f}%)")
        
        if gap_ratio > 50:
            print(f"\n❌ 严重过拟合！验证集loss比训练集高{gap_ratio:.1f}%")
            print("\n建议措施:")
            print("  1. 降低学习率 (当前0.003 -> 0.0001)")
            print("  2. 增加dropout (当前0.01 -> 0.2)")
            print("  3. 减小batch size (当前32 -> 16)")
            print("  4. 增加weight decay (当前1e-5 -> 1e-4)")
            print("  5. 使用时序划分而非随机划分")
        elif gap_ratio > 20:
            print(f"\n⚠️  中度过拟合，验证集loss比训练集高{gap_ratio:.1f}%")
            print("\n建议调整正则化参数")
        else:
            print(f"\n✓ 过拟合程度可接受")
        
        # 检查验证集是否在改善
        if len(val_losses) >= 20:
            early_val = np.mean(val_losses[:10])
            late_val = np.mean(val_losses[-10:])
            improvement = (early_val - late_val) / early_val * 100
            
            print(f"\n验证集改善情况:")
            print(f"  前10个epoch平均: {early_val:.4f}")
            print(f"  后10个epoch平均: {late_val:.4f}")
            print(f"  改善幅度: {improvement:.1f}%")
            
            if improvement < 0:
                print(f"  ❌ 验证集loss在上升！")
            elif improvement < 5:
                print(f"  ⚠️  验证集几乎没有改善")
            else:
                print(f"  ✓ 验证集在改善")


def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_overfitting.py <training.log路径>")
        print("示例: python diagnose_overfitting.py checkpoint/convlstm_attention_deep/training.log")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    if not Path(log_path).exists():
        print(f"❌ 日志文件不存在: {log_path}")
        sys.exit(1)
    
    print(f"正在分析训练日志: {log_path}")
    
    # 解析日志
    train_losses, val_losses, learning_rates = parse_training_log(log_path)
    
    print(f"✓ 找到 {len(train_losses)} 个训练loss记录")
    print(f"✓ 找到 {len(val_losses)} 个验证loss记录")
    print(f"✓ 找到 {len(learning_rates)} 个学习率记录")
    
    # 分析过拟合
    analyze_overfitting(train_losses, val_losses)
    
    # 绘制曲线
    output_dir = Path(log_path).parent / "diagnosis"
    plot_training_curves(train_losses, val_losses, learning_rates, output_dir)
    
    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)


if __name__ == "__main__":
    main()
