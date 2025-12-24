#!/usr/bin/env python3
"""
So sanh ket qua V7 vs V7.1

Load artifacts tu ca V7 va V7.1, so sanh metrics:
- Win rates (buy, sell, combined)
- Expectancy
- Profitability (total return, profit factor, Sharpe)
- Model stability
"""

import os
import torch
import numpy as np
import pandas as pd


def load_artifact(artifact_path):
    """Load artifact file"""
    if not os.path.exists(artifact_path):
        return None
    # PyTorch 2.6+ requires weights_only=False for loading artifacts with numpy objects
    return torch.load(artifact_path, map_location='cpu', weights_only=False)


def compare_metrics(v7_artifact, v7_1_artifact):
    """So sanh metrics giua V7 va V7.1"""
    print(f"\n{'='*60}")
    print("So sanh V7 vs V7.1")
    print(f"{'='*60}")
    
    if v7_artifact is None:
        print("  V7 artifact khong tim thay. Chay V7 truoc.")
        return
    
    if v7_1_artifact is None:
        print("  V7.1 artifact khong tim thay. Chay V7.1 truoc.")
        return
    
    # Extract metrics
    v7_test_metrics = v7_artifact.get('test_metrics', {})
    v7_test_profit = v7_artifact.get('test_profitability', {})
    v7_pretrain = v7_artifact.get('pretrain_metrics', {})
    v7_finetune = v7_artifact.get('finetune_metrics', {})
    
    v7_1_test_metrics = v7_1_artifact.get('test_metrics', {})
    v7_1_test_profit = v7_1_artifact.get('test_profitability', {})
    v7_1_pretrain = v7_1_artifact.get('pretrain_metrics', {})
    v7_1_finetune = v7_1_artifact.get('finetune_metrics', {})
    
    # Training metrics
    print(f"\n{'='*60}")
    print("Training Metrics:")
    print(f"{'='*60}")
    print(f"\nPretrain:")
    print(f"  V7  - Best Val Loss: {v7_pretrain.get('best_val_loss', 'N/A'):.6f}" if v7_pretrain.get('best_val_loss') else "  V7  - Best Val Loss: N/A")
    print(f"  V7.1 - Best Val Loss: {v7_1_pretrain.get('best_val_loss', 'N/A'):.6f}" if v7_1_pretrain.get('best_val_loss') else "  V7.1 - Best Val Loss: N/A")
    
    print(f"\nFine-tune:")
    print(f"  V7  - Best Val Loss: {v7_finetune.get('best_val_loss', 'N/A'):.6f}" if v7_finetune.get('best_val_loss') else "  V7  - Best Val Loss: N/A")
    print(f"  V7.1 - Best Val Loss: {v7_1_finetune.get('best_val_loss', 'N/A'):.6f}" if v7_1_finetune.get('best_val_loss') else "  V7.1 - Best Val Loss: N/A")
    
    # Test metrics
    print(f"\n{'='*60}")
    print("Test Metrics (2025):")
    print(f"{'='*60}")
    
    metrics_to_compare = [
        ('buy_wr', 'Buy Win Rate'),
        ('sell_wr', 'Sell Win Rate'),
        ('combined_wr', 'Combined Win Rate'),
        ('buy_expectancy', 'Buy Expectancy'),
        ('sell_expectancy', 'Sell Expectancy'),
        ('coverage', 'Coverage'),
        ('rmse', 'RMSE'),
        ('mae', 'MAE')
    ]
    
    for metric_key, metric_name in metrics_to_compare:
        v7_val = v7_test_metrics.get(metric_key, 0)
        v7_1_val = v7_1_test_metrics.get(metric_key, 0)
        diff = v7_1_val - v7_val
        diff_pct = (diff / v7_val * 100) if v7_val != 0 else 0
        
        if 'wr' in metric_key or 'coverage' in metric_key:
            print(f"  {metric_name}:")
            print(f"    V7:   {v7_val:.1%}")
            print(f"    V7.1: {v7_1_val:.1%}")
            print(f"    Diff: {diff:+.1%} ({diff_pct:+.1f}%)")
        elif 'expectancy' in metric_key:
            print(f"  {metric_name}:")
            print(f"    V7:   {v7_val:.4f}")
            print(f"    V7.1: {v7_1_val:.4f}")
            print(f"    Diff: {diff:+.4f} ({diff_pct:+.1f}%)")
        else:
            print(f"  {metric_name}:")
            print(f"    V7:   {v7_val:.4f}")
            print(f"    V7.1: {v7_1_val:.4f}")
            print(f"    Diff: {diff:+.4f} ({diff_pct:+.1f}%)")
    
    # Profitability metrics
    print(f"\n{'='*60}")
    print("Profitability Metrics:")
    print(f"{'='*60}")
    
    profit_metrics = [
        ('total_return', 'Total Return'),
        ('profit_factor', 'Profit Factor'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('max_drawdown', 'Max Drawdown'),
        ('is_profitable', 'Is Profitable')
    ]
    
    for metric_key, metric_name in profit_metrics:
        v7_val = v7_test_profit.get(metric_key, 0)
        v7_1_val = v7_1_test_profit.get(metric_key, 0)
        
        if metric_key == 'is_profitable':
            print(f"  {metric_name}:")
            print(f"    V7:   {v7_val}")
            print(f"    V7.1: {v7_1_val}")
        elif metric_key == 'total_return' or metric_key == 'max_drawdown':
            diff = v7_1_val - v7_val
            print(f"  {metric_name}:")
            print(f"    V7:   {v7_val:.2%}")
            print(f"    V7.1: {v7_1_val:.2%}")
            print(f"    Diff: {diff:+.2%}")
        else:
            diff = v7_1_val - v7_val
            print(f"  {metric_name}:")
            print(f"    V7:   {v7_val:.2f}")
            print(f"    V7.1: {v7_1_val:.2f}")
            print(f"    Diff: {diff:+.2f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    
    v7_combined_wr = v7_test_metrics.get('combined_wr', 0)
    v7_1_combined_wr = v7_1_test_metrics.get('combined_wr', 0)
    v7_total_return = v7_test_profit.get('total_return', 0)
    v7_1_total_return = v7_1_test_profit.get('total_return', 0)
    v7_profit_factor = v7_test_profit.get('profit_factor', 0)
    v7_1_profit_factor = v7_1_test_profit.get('profit_factor', 0)
    
    print(f"\n  Combined Win Rate:")
    print(f"    V7:   {v7_combined_wr:.1%}")
    print(f"    V7.1: {v7_1_combined_wr:.1%}")
    print(f"    {'V7.1 tốt hơn' if v7_1_combined_wr > v7_combined_wr else 'V7 tốt hơn' if v7_combined_wr > v7_1_combined_wr else 'Bằng nhau'}")
    
    print(f"\n  Total Return:")
    print(f"    V7:   {v7_total_return:.2%}")
    print(f"    V7.1: {v7_1_total_return:.2%}")
    print(f"    {'V7.1 tốt hơn' if v7_1_total_return > v7_total_return else 'V7 tốt hơn' if v7_total_return > v7_1_total_return else 'Bằng nhau'}")
    
    print(f"\n  Profit Factor:")
    print(f"    V7:   {v7_profit_factor:.2f}")
    print(f"    V7.1: {v7_1_profit_factor:.2f}")
    print(f"    {'V7.1 tốt hơn' if v7_1_profit_factor > v7_profit_factor else 'V7 tốt hơn' if v7_profit_factor > v7_1_profit_factor else 'Bằng nhau'}")
    
    print(f"\n  Data Leakage:")
    print(f"    V7:   Có (pretrain 2015-2025 bao gồm test 2025)")
    print(f"    V7.1: Không (pretrain 2015-2020, finetune 2021-2024, test 2025 tách biệt)")
    
    print(f"\n  Generalization:")
    print(f"    V7:   Thấp hơn (có thể overfit do leakage)")
    print(f"    V7.1: Cao hơn (no leakage, out-of-sample thực sự)")
    
    # Save comparison to CSV
    comparison_data = {
        'Metric': [],
        'V7': [],
        'V7.1': [],
        'Difference': [],
        'Difference %': []
    }
    
    for metric_key, metric_name in metrics_to_compare:
        v7_val = v7_test_metrics.get(metric_key, 0)
        v7_1_val = v7_1_test_metrics.get(metric_key, 0)
        diff = v7_1_val - v7_val
        diff_pct = (diff / v7_val * 100) if v7_val != 0 else 0
        
        comparison_data['Metric'].append(metric_name)
        comparison_data['V7'].append(v7_val)
        comparison_data['V7.1'].append(v7_1_val)
        comparison_data['Difference'].append(diff)
        comparison_data['Difference %'].append(diff_pct)
    
    for metric_key, metric_name in profit_metrics:
        if metric_key == 'is_profitable':
            continue
        v7_val = v7_test_profit.get(metric_key, 0)
        v7_1_val = v7_1_test_profit.get(metric_key, 0)
        diff = v7_1_val - v7_val
        diff_pct = (diff / v7_val * 100) if v7_val != 0 else 0
        
        comparison_data['Metric'].append(metric_name)
        comparison_data['V7'].append(v7_val)
        comparison_data['V7.1'].append(v7_1_val)
        comparison_data['Difference'].append(diff)
        comparison_data['Difference %'].append(diff_pct)
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv('v7_v7_1_comparison.csv', index=False)
    print(f"\n  Comparison saved to: v7_v7_1_comparison.csv")


def main():
    # Load artifacts
    v7_artifact_path = 'nvda_lstm_v7_pretrain_finetune_artifact.pth'
    v7_1_artifact_path = 'nvda_lstm_v7_1_artifact.pth'
    
    print("Dang load artifacts...")
    v7_artifact = load_artifact(v7_artifact_path)
    v7_1_artifact = load_artifact(v7_1_artifact_path)
    
    if v7_artifact is None:
        print(f"Khong tim thay V7 artifact: {v7_artifact_path}")
        print("Vui long chay V7 truoc.")
        return
    
    if v7_1_artifact is None:
        print(f"Khong tim thay V7.1 artifact: {v7_1_artifact_path}")
        print("Vui long chay V7.1 truoc.")
        return
    
    # Compare
    compare_metrics(v7_artifact, v7_1_artifact)


if __name__ == '__main__':
    main()

