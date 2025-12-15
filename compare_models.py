#!/usr/bin/env python3
"""对比单区域和上下游双区域模型的预报准确率。

此脚本用于评估两个ConvLSTM模型在整个测试数据集上的性能，特别关注：
1. 整体预报准确率（RMSE, MAE, R²等）
2. 规模降雨预报准确率（不同降雨强度阈值下的准确率）
3. 空间分布误差分析
4. 时间序列性能对比

使用示例:
    # 基本对比
    python compare_models.py \\
        --model1-checkpoint checkpoints/baseline/best_model.pt \\
        --model1-normalizer checkpoints/baseline/normalizer.pkl \\
        --model2-checkpoint checkpoints/with_upstream/best_model.pt \\
        --model2-normalizer checkpoints/with_upstream/normalizer.pkl \\
        --model2-upstream \\
        --data data/test_data.nc \\
        --output-dir outputs/comparison
    
    # 指定时间范围
    python compare_models.py \\
        --model1-checkpoint checkpoints/baseline/best_model.pt \\
        --model1-normalizer checkpoints/baseline/normalizer.pkl \\
        --model2-checkpoint checkpoints/with_upstream/best_model.pt \\
        --model2-normalizer checkpoints/with_upstream/normalizer.pkl \\
        --model2-upstream \\
        --data data/test_data.nc \\
        --output-dir outputs/comparison \\
        --start-time 2020-01-01T00:00:00 \\
        --end-time 2020-12-31T23:00:00
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from run_inference_convlstm import (
    load_trained_model_auto,
    load_data,
    setup_logging
)
from convlstm.inference import load_normalizer, predict_batch
from convlstm.data import RegionConfig


def calculate_metrics(
    predictions: xr.Dataset,
    ground_truth: xr.Dataset
) -> Dict[str, float]:
    """计算预报准确率指标。
    
    Args:
        predictions: 预测数据集
        ground_truth: 真实数据集
        
    Returns:
        包含各种指标的字典
    """
    # 提取数据并展平
    pred_values = predictions.precipitation.values.flatten()
    truth_values = ground_truth.precipitation.values.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(pred_values) | np.isnan(truth_values))
    pred_values = pred_values[mask]
    truth_values = truth_values[mask]
    
    # 计算基本指标
    rmse = np.sqrt(mean_squared_error(truth_values, pred_values))
    mae = mean_absolute_error(truth_values, pred_values)
    r2 = r2_score(truth_values, pred_values)
    
    # 计算相关系数
    correlation, _ = stats.pearsonr(truth_values, pred_values)
    
    # 计算偏差
    bias = np.mean(pred_values - truth_values)
    
    # 计算相对误差
    relative_error = np.mean(np.abs(pred_values - truth_values) / (truth_values + 1e-6)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Correlation': correlation,
        'Bias': bias,
        'Relative_Error_%': relative_error
    }


def calculate_rainfall_category_metrics(
    predictions: xr.Dataset,
    ground_truth: xr.Dataset,
    thresholds: List[float] = [0.1, 1.0, 10.0, 25.0, 50.0, 100.0]
) -> pd.DataFrame:
    """计算不同降雨强度阈值下的准确率指标。
    
    Args:
        predictions: 预测数据集
        ground_truth: 真实数据集
        thresholds: 降雨强度阈值列表（mm）
        
    Returns:
        包含各阈值下指标的DataFrame
    """
    pred_values = predictions.precipitation.values.flatten()
    truth_values = ground_truth.precipitation.values.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(pred_values) | np.isnan(truth_values))
    pred_values = pred_values[mask]
    truth_values = truth_values[mask]
    
    results = []
    
    for threshold in thresholds:
        # 二分类：是否超过阈值
        pred_binary = (pred_values >= threshold).astype(int)
        truth_binary = (truth_values >= threshold).astype(int)
        
        # 计算混淆矩阵
        tp = np.sum((pred_binary == 1) & (truth_binary == 1))
        tn = np.sum((pred_binary == 0) & (truth_binary == 0))
        fp = np.sum((pred_binary == 1) & (truth_binary == 0))
        fn = np.sum((pred_binary == 0) & (truth_binary == 1))
        
        # 计算指标
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # POD (Probability of Detection) 和 FAR (False Alarm Ratio)
        pod = recall
        far = fp / (tp + fp) if (tp + fp) > 0 else 0
        
        # CSI (Critical Success Index) / TS (Threat Score)
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        results.append({
            'Threshold_mm': threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall_POD': recall,
            'F1_Score': f1,
            'FAR': far,
            'CSI_TS': csi,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        })
    
    return pd.DataFrame(results)


def plot_comparison_metrics(
    model1_metrics: Dict[str, float],
    model2_metrics: Dict[str, float],
    model1_name: str,
    model2_name: str,
    output_dir: Path
):
    """绘制两个模型的整体指标对比图。"""
    metrics_to_plot = ['RMSE', 'MAE', 'R²', 'Correlation', 'Bias']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [model1_metrics[metric], model2_metrics[metric]]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar([model1_name, model2_name], values, color=colors, alpha=0.7)
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_rainfall_category_comparison(
    model1_df: pd.DataFrame,
    model2_df: pd.DataFrame,
    model1_name: str,
    model2_name: str,
    output_dir: Path
):
    """绘制不同降雨强度阈值下的准确率对比图。"""
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall_POD', 'F1_Score', 'CSI_TS']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        ax.plot(model1_df['Threshold_mm'], model1_df[metric], 
               marker='o', linewidth=2, markersize=8, label=model1_name, color='#3498db')
        ax.plot(model2_df['Threshold_mm'], model2_df[metric], 
               marker='s', linewidth=2, markersize=8, label=model2_name, color='#e74c3c')
        
        ax.set_xlabel('Rainfall Threshold (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} vs Rainfall Threshold', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Hide unused subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rainfall_category_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_comparison(
    predictions1: xr.Dataset,
    predictions2: xr.Dataset,
    ground_truth: xr.Dataset,
    model1_name: str,
    model2_name: str,
    output_dir: Path
):
    """绘制预测值与真实值的散点图对比。"""
    # 提取数据
    pred1_values = predictions1.precipitation.values.flatten()
    pred2_values = predictions2.precipitation.values.flatten()
    truth_values = ground_truth.precipitation.values.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(pred1_values) | np.isnan(pred2_values) | np.isnan(truth_values))
    pred1_values = pred1_values[mask]
    pred2_values = pred2_values[mask]
    truth_values = truth_values[mask]
    
    # 为了可视化，随机采样部分数据点
    if len(truth_values) > 10000:
        indices = np.random.choice(len(truth_values), 10000, replace=False)
        pred1_values = pred1_values[indices]
        pred2_values = pred2_values[indices]
        truth_values = truth_values[indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Model 1 scatter plot
    ax = axes[0]
    ax.scatter(truth_values, pred1_values, alpha=0.3, s=10, c='#3498db')
    ax.plot([0, truth_values.max()], [0, truth_values.max()], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Observed Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model1_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Model 2 scatter plot
    ax = axes[1]
    ax.scatter(truth_values, pred2_values, alpha=0.3, s=10, c='#e74c3c')
    ax.plot([0, truth_values.max()], [0, truth_values.max()], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Observed Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model2_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_comparison_report(
    model1_metrics: Dict[str, float],
    model2_metrics: Dict[str, float],
    model1_rainfall_df: pd.DataFrame,
    model2_rainfall_df: pd.DataFrame,
    model1_name: str,
    model2_name: str,
    output_dir: Path
):
    """生成详细的对比报告。"""
    report_path = output_dir / 'comparison_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ConvLSTM 模型对比报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"模型1: {model1_name}\n")
        f.write(f"模型2: {model2_name}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("1. 整体预报准确率指标\n")
        f.write("=" * 80 + "\n\n")
        
        # 整体指标对比表
        f.write(f"{'指标':<20} {model1_name:<20} {model2_name:<20} {'改进%':<15}\n")
        f.write("-" * 80 + "\n")
        
        for metric in ['RMSE', 'MAE', 'R²', 'Correlation', 'Bias', 'Relative_Error_%']:
            val1 = model1_metrics[metric]
            val2 = model2_metrics[metric]
            
            # 计算改进百分比（对于RMSE、MAE、Bias，越小越好；对于R²、Correlation，越大越好）
            if metric in ['RMSE', 'MAE', 'Bias', 'Relative_Error_%']:
                improvement = ((val1 - val2) / abs(val1)) * 100 if val1 != 0 else 0
            else:
                improvement = ((val2 - val1) / abs(val1)) * 100 if val1 != 0 else 0
            
            f.write(f"{metric:<20} {val1:<20.4f} {val2:<20.4f} {improvement:>+14.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("2. 规模降雨预报准确率（不同降雨强度阈值）\n")
        f.write("=" * 80 + "\n\n")
        
        # 降雨分类指标对比
        f.write(f"阈值(mm)  指标          {model1_name:<15} {model2_name:<15} 改进%\n")
        f.write("-" * 80 + "\n")
        
        for _, row1 in model1_rainfall_df.iterrows():
            threshold = row1['Threshold_mm']
            row2 = model2_rainfall_df[model2_rainfall_df['Threshold_mm'] == threshold].iloc[0]
            
            f.write(f"\n{threshold:>8.1f}mm:\n")
            
            for metric in ['Accuracy', 'Precision', 'Recall_POD', 'F1_Score', 'CSI_TS']:
                val1 = row1[metric]
                val2 = row2[metric]
                improvement = ((val2 - val1) / val1) * 100 if val1 != 0 else 0
                
                f.write(f"          {metric:<12} {val1:<15.4f} {val2:<15.4f} {improvement:>+7.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("3. 总结\n")
        f.write("=" * 80 + "\n\n")
        
        # 自动生成总结
        rmse_improvement = ((model1_metrics['RMSE'] - model2_metrics['RMSE']) / model1_metrics['RMSE']) * 100
        r2_improvement = ((model2_metrics['R²'] - model1_metrics['R²']) / abs(model1_metrics['R²'])) * 100
        
        if rmse_improvement > 0:
            f.write(f"✓ {model2_name} 的RMSE比{model1_name}降低了 {rmse_improvement:.2f}%\n")
        else:
            f.write(f"✗ {model2_name} 的RMSE比{model1_name}增加了 {abs(rmse_improvement):.2f}%\n")
        
        if r2_improvement > 0:
            f.write(f"✓ {model2_name} 的R²比{model1_name}提高了 {r2_improvement:.2f}%\n")
        else:
            f.write(f"✗ {model2_name} 的R²比{model1_name}降低了 {abs(r2_improvement):.2f}%\n")
        
        # 分析不同降雨强度下的表现
        f.write("\nRainfall Intensity Analysis:\n")
        for threshold in [0.1, 10.0, 50.0]:
            row1 = model1_rainfall_df[model1_rainfall_df['Threshold_mm'] == threshold].iloc[0]
            row2 = model2_rainfall_df[model2_rainfall_df['Threshold_mm'] == threshold].iloc[0]
            
            csi_improvement = ((row2['CSI_TS'] - row1['CSI_TS']) / row1['CSI_TS']) * 100 if row1['CSI_TS'] != 0 else 0
            
            if threshold == 0.1:
                category = "Light Rain"
            elif threshold == 10.0:
                category = "Moderate Rain"
            else:
                category = "Heavy Rain"
            
            if csi_improvement > 0:
                f.write(f"✓ {category} (≥{threshold}mm): CSI improved by {csi_improvement:.2f}%\n")
            else:
                f.write(f"✗ {category} (≥{threshold}mm): CSI decreased by {abs(csi_improvement):.2f}%\n")
    
    print(f"\n详细报告已保存至: {report_path}")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对比两个ConvLSTM模型的预报准确率",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 模型1配置
    model1_group = parser.add_argument_group('模型1配置（通常是基线模型）')
    model1_group.add_argument(
        '--model1-checkpoint',
        type=str,
        required=True,
        help='模型1的checkpoint路径'
    )
    model1_group.add_argument(
        '--model1-normalizer',
        type=str,
        required=True,
        help='模型1的normalizer路径'
    )
    model1_group.add_argument(
        '--model1-name',
        type=str,
        default='Single-Region Model',
        help='模型1的名称（用于报告和图表）'
    )
    model1_group.add_argument(
        '--model1-upstream',
        action='store_true',
        help='模型1是否使用上游区域'
    )
    model1_group.add_argument(
        '--model1-type',
        type=str,
        default=None,
        choices=['shallow', 'deep', 'dual_stream', 'dual_stream_deep'],
        help='模型1架构类型（可选，自动检测）'
    )
    
    # 模型2配置
    model2_group = parser.add_argument_group('模型2配置（通常是改进模型）')
    model2_group.add_argument(
        '--model2-checkpoint',
        type=str,
        required=True,
        help='模型2的checkpoint路径'
    )
    model2_group.add_argument(
        '--model2-normalizer',
        type=str,
        required=True,
        help='模型2的normalizer路径'
    )
    model2_group.add_argument(
        '--model2-name',
        type=str,
        default='Dual-Stream Model',
        help='模型2的名称（用于报告和图表）'
    )
    model2_group.add_argument(
        '--model2-upstream',
        action='store_true',
        help='模型2是否使用上游区域'
    )
    model2_group.add_argument(
        '--model2-type',
        type=str,
        default=None,
        choices=['shallow', 'deep', 'dual_stream', 'dual_stream_deep'],
        help='模型2架构类型（可选，自动检测）'
    )
    
    # 数据配置
    data_group = parser.add_argument_group('数据配置')
    data_group.add_argument(
        '--data',
        type=str,
        required=True,
        help='测试数据路径（NetCDF格式）'
    )
    data_group.add_argument(
        '--start-time',
        type=str,
        default=None,
        help='开始时间（ISO格式：YYYY-MM-DDTHH:MM:SS）'
    )
    data_group.add_argument(
        '--end-time',
        type=str,
        default=None,
        help='结束时间（ISO格式：YYYY-MM-DDTHH:MM:SS）'
    )
    
    # 区域配置
    region_group = parser.add_argument_group('区域配置')
    region_group.add_argument('--downstream-lat-min', type=float, default=25.0)
    region_group.add_argument('--downstream-lat-max', type=float, default=40.0)
    region_group.add_argument('--downstream-lon-min', type=float, default=110.0)
    region_group.add_argument('--downstream-lon-max', type=float, default=125.0)
    region_group.add_argument('--upstream-lat-min', type=float, default=25.0)
    region_group.add_argument('--upstream-lat-max', type=float, default=50.0)
    region_group.add_argument('--upstream-lon-min', type=float, default=70.0)
    region_group.add_argument('--upstream-lon-max', type=float, default=110.0)
    
    # 推理配置
    inference_group = parser.add_argument_group('推理配置')
    inference_group.add_argument('--window-size', type=int, default=6)
    inference_group.add_argument('--target-offset', type=int, default=1)
    inference_group.add_argument('--batch-size', type=int, default=8)
    
    # 降雨阈值配置
    threshold_group = parser.add_argument_group('降雨阈值配置')
    threshold_group.add_argument(
        '--rainfall-thresholds',
        type=float,
        nargs='+',
        default=[0.1, 1.0, 10.0, 25.0, 50.0, 100.0],
        help='降雨强度阈值列表（mm）'
    )
    
    # 输出配置
    output_group = parser.add_argument_group('输出配置')
    output_group.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='输出目录'
    )
    output_group.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='计算设备'
    )
    output_group.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR']
    )
    
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir, args.log_level)
    
    logger.info("=" * 80)
    logger.info("ConvLSTM 模型对比分析")
    logger.info("=" * 80)
    logger.info(f"模型1: {args.model1_name}")
    logger.info(f"  Checkpoint: {args.model1_checkpoint}")
    logger.info(f"  使用上游区域: {args.model1_upstream}")
    logger.info(f"模型2: {args.model2_name}")
    logger.info(f"  Checkpoint: {args.model2_checkpoint}")
    logger.info(f"  使用上游区域: {args.model2_upstream}")
    logger.info(f"测试数据: {args.data}")
    logger.info(f"输出目录: {output_dir}")
    
    # 设置设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("\n" + "=" * 80)
    logger.info("加载测试数据...")
    logger.info("=" * 80)
    
    try:
        data = load_data(
            data_path=args.data,
            logger=logger,
            start_time=args.start_time,
            end_time=args.end_time
        )
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        sys.exit(1)
    
    # 创建区域配置
    region_config = RegionConfig(
        downstream_lat_min=args.downstream_lat_min,
        downstream_lat_max=args.downstream_lat_max,
        downstream_lon_min=args.downstream_lon_min,
        downstream_lon_max=args.downstream_lon_max,
        upstream_lat_min=args.upstream_lat_min,
        upstream_lat_max=args.upstream_lat_max,
        upstream_lon_min=args.upstream_lon_min,
        upstream_lon_max=args.upstream_lon_max
    )
    
    # 加载模型1
    logger.info("\n" + "=" * 80)
    logger.info(f"加载模型1: {args.model1_name}")
    logger.info("=" * 80)
    
    try:
        model1, checkpoint1, model1_type = load_trained_model_auto(
            args.model1_checkpoint,
            device=device,
            model_type_override=args.model1_type
        )
        normalizer1 = load_normalizer(args.model1_normalizer)
        logger.info(f"模型1类型: {model1_type}")
        logger.info(f"训练轮次: {checkpoint1['epoch']}")
        logger.info(f"最佳验证损失: {checkpoint1['best_val_loss']:.4f}")
    except Exception as e:
        logger.error(f"加载模型1失败: {e}")
        sys.exit(1)
    
    # 加载模型2
    logger.info("\n" + "=" * 80)
    logger.info(f"加载模型2: {args.model2_name}")
    logger.info("=" * 80)
    
    try:
        model2, checkpoint2, model2_type = load_trained_model_auto(
            args.model2_checkpoint,
            device=device,
            model_type_override=args.model2_type
        )
        normalizer2 = load_normalizer(args.model2_normalizer)
        logger.info(f"模型2类型: {model2_type}")
        logger.info(f"训练轮次: {checkpoint2['epoch']}")
        logger.info(f"最佳验证损失: {checkpoint2['best_val_loss']:.4f}")
    except Exception as e:
        logger.error(f"加载模型2失败: {e}")
        sys.exit(1)
    
    # 生成预测 - 模型1
    logger.info("\n" + "=" * 80)
    logger.info(f"生成模型1预测...")
    logger.info("=" * 80)
    
    try:
        predictions1 = predict_batch(
            model=model1,
            input_data=data,
            normalizer=normalizer1,
            region_config=region_config,
            window_size=args.window_size,
            target_offset=args.target_offset,
            include_upstream=args.model1_upstream,
            batch_size=args.batch_size,
            device=device
        )
        logger.info(f"生成了 {len(predictions1.time)} 个预测")
    except Exception as e:
        logger.error(f"模型1预测失败: {e}")
        sys.exit(1)
    
    # 生成预测 - 模型2
    logger.info("\n" + "=" * 80)
    logger.info(f"生成模型2预测...")
    logger.info("=" * 80)
    
    try:
        predictions2 = predict_batch(
            model=model2,
            input_data=data,
            normalizer=normalizer2,
            region_config=region_config,
            window_size=args.window_size,
            target_offset=args.target_offset,
            include_upstream=args.model2_upstream,
            batch_size=args.batch_size,
            device=device
        )
        logger.info(f"生成了 {len(predictions2.time)} 个预测")
    except Exception as e:
        logger.error(f"模型2预测失败: {e}")
        sys.exit(1)
    
    # 提取真实值
    logger.info("\n" + "=" * 80)
    logger.info("提取真实值...")
    logger.info("=" * 80)
    
    downstream_data = data.sel(
        lat=slice(region_config.downstream_lat_min, region_config.downstream_lat_max),
        lon=slice(region_config.downstream_lon_min, region_config.downstream_lon_max)
    )
    
    # 对齐时间维度
    ground_truth = downstream_data.sel(time=predictions1.time)
    logger.info(f"真实值时间步数: {len(ground_truth.time)}")
    
    # 计算整体指标
    logger.info("\n" + "=" * 80)
    logger.info("计算整体预报准确率指标...")
    logger.info("=" * 80)
    
    model1_metrics = calculate_metrics(predictions1, ground_truth)
    model2_metrics = calculate_metrics(predictions2, ground_truth)
    
    logger.info(f"\n{args.model1_name} 指标:")
    for metric, value in model1_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"\n{args.model2_name} 指标:")
    for metric, value in model2_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # 计算降雨分类指标
    logger.info("\n" + "=" * 80)
    logger.info("计算规模降雨预报准确率...")
    logger.info("=" * 80)
    
    model1_rainfall_df = calculate_rainfall_category_metrics(
        predictions1, ground_truth, args.rainfall_thresholds
    )
    model2_rainfall_df = calculate_rainfall_category_metrics(
        predictions2, ground_truth, args.rainfall_thresholds
    )
    
    logger.info(f"\n{args.model1_name} 降雨分类指标:")
    logger.info(model1_rainfall_df.to_string())
    
    logger.info(f"\n{args.model2_name} 降雨分类指标:")
    logger.info(model2_rainfall_df.to_string())
    
    # 保存结果到CSV
    model1_rainfall_df.to_csv(output_dir / f'{args.model1_name.replace(" ", "_")}_rainfall_metrics.csv', index=False)
    model2_rainfall_df.to_csv(output_dir / f'{args.model2_name.replace(" ", "_")}_rainfall_metrics.csv', index=False)
    
    # 生成可视化
    logger.info("\n" + "=" * 80)
    logger.info("生成对比图表...")
    logger.info("=" * 80)
    
    plot_comparison_metrics(
        model1_metrics, model2_metrics,
        args.model1_name, args.model2_name,
        output_dir
    )
    logger.info("✓ 整体指标对比图已保存")
    
    plot_rainfall_category_comparison(
        model1_rainfall_df, model2_rainfall_df,
        args.model1_name, args.model2_name,
        output_dir
    )
    logger.info("✓ 降雨分类对比图已保存")
    
    plot_scatter_comparison(
        predictions1, predictions2, ground_truth,
        args.model1_name, args.model2_name,
        output_dir
    )
    logger.info("✓ 散点图对比已保存")
    
    # 生成详细报告
    logger.info("\n" + "=" * 80)
    logger.info("生成对比报告...")
    logger.info("=" * 80)
    
    generate_comparison_report(
        model1_metrics, model2_metrics,
        model1_rainfall_df, model2_rainfall_df,
        args.model1_name, args.model2_name,
        output_dir
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("对比分析完成！")
    logger.info("=" * 80)
    logger.info(f"所有结果已保存至: {output_dir}")
    logger.info(f"  - 对比报告: {output_dir / 'comparison_report.txt'}")
    logger.info(f"  - 整体指标图: {output_dir / 'overall_metrics_comparison.png'}")
    logger.info(f"  - 降雨分类图: {output_dir / 'rainfall_category_comparison.png'}")
    logger.info(f"  - 散点对比图: {output_dir / 'scatter_comparison.png'}")
    logger.info(f"  - CSV数据: {output_dir / '*.csv'}")


if __name__ == "__main__":
    main()
