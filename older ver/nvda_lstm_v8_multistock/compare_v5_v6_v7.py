#!/usr/bin/env python3
"""
So sanh ket qua V5, V6, V7

Load artifacts tu V5, V6, V7, so sanh metrics:
- Win rates (buy, sell, combined)
- Expectancy
- Profitability (total return, profit factor, Sharpe)
- Model stability
- Training metrics
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_artifact(artifact_path):
    """Load artifact file"""
    if not os.path.exists(artifact_path):
        return None
    # PyTorch 2.6+ requires weights_only=False for loading artifacts with numpy objects
    try:
        return torch.load(artifact_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Loi khi load {artifact_path}: {e}")
        return None


def compare_three_versions():
    """So sanh metrics giua V5, V6, V7"""
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths to artifacts
    v5_artifact = os.path.join(workspace_root, 'nvda_lstm_v5_multistock', 'nvda_lstm_v5_artifact.pth')
    v6_artifact = os.path.join(workspace_root, 'nvda_lstm_v6_multistock', 'nvda_lstm_v6_artifact.pth')
    v7_artifact = os.path.join(workspace_root, 'nvda_lstm_v7_multistock', 'nvda_lstm_v7_multistock_artifact.pth')
    
    # Load artifacts
    v5_data = load_artifact(v5_artifact)
    v6_data = load_artifact(v6_artifact)
    v7_data = load_artifact(v7_artifact)
    
    print(f"\n{'='*80}")
    print("SO SANH KET QUA V5 vs V6 vs V7")
    print(f"{'='*80}")
    
    if v5_data is None:
        print("  V5 artifact khong tim thay. Chay V5 truoc.")
    if v6_data is None:
        print("  V6 artifact khong tim thay. Chay V6 truoc.")
    if v7_data is None:
        print("  V7 artifact khong tim thay. Chay V7 truoc.")
    
    if v5_data is None and v6_data is None and v7_data is None:
        print("\nKhong co artifact nao de so sanh. Vui long chay cac versions truoc.")
        return
    
    # Extract metrics
    v5_metrics = v5_data.get('metrics', {}) if v5_data else {}
    v6_metrics = v6_data.get('metrics', {}) if v6_data else {}
    v7_test_metrics = v7_data.get('test_metrics', {}) if v7_data else {}
    
    v5_pretrain = v5_data.get('pretrain_metrics', {}) if v5_data else {}
    v6_pretrain = v6_data.get('pretrain_metrics', {}) if v6_data else {}
    v7_pretrain = v7_data.get('pretrain_metrics', {}) if v7_data else {}
    
    v5_profit = v5_data.get('profitability', {}) if v5_data else {}
    v6_profit = v6_data.get('profitability', {}) if v6_data else {}
    v7_profit = v7_data.get('test_profitability', {}) if v7_data else {}
    
    # Training metrics
    print(f"\n{'='*80}")
    print("Training Metrics:")
    print(f"{'='*80}")
    print(f"\nPretrain Best Val Loss:")
    if v5_pretrain:
        print(f"  V5: {v5_pretrain.get('best_val_loss', 'N/A'):.6f}" if v5_pretrain.get('best_val_loss') else "  V5: N/A")
    if v6_pretrain:
        print(f"  V6: {v6_pretrain.get('best_val_loss', 'N/A'):.6f}" if v6_pretrain.get('best_val_loss') else "  V6: N/A")
    if v7_pretrain:
        print(f"  V7: {v7_pretrain.get('best_val_loss', 'N/A'):.6f}" if v7_pretrain.get('best_val_loss') else "  V7: N/A")
    
    # Test metrics
    print(f"\n{'='*80}")
    print("Test Metrics:")
    print(f"{'='*80}")
    
    metrics_to_compare = [
        ('buy_wr', 'Buy Win Rate', '%'),
        ('sell_wr', 'Sell Win Rate', '%'),
        ('combined_wr', 'Combined Win Rate', '%'),
        ('buy_expectancy', 'Buy Expectancy', 'abs'),
        ('sell_expectancy', 'Sell Expectancy', 'abs'),
        ('coverage', 'Coverage', '%'),
        ('rmse', 'RMSE', 'abs'),
        ('mae', 'MAE', 'abs')
    ]
    
    comparison_data = []
    
    for metric_key, metric_name, fmt in metrics_to_compare:
        v5_val = v5_metrics.get(metric_key, 0) if v5_data else 0
        v6_val = v6_metrics.get(metric_key, 0) if v6_data else 0
        v7_val = v7_test_metrics.get(metric_key, 0) if v7_data else 0
        
        comparison_data.append({
            'Metric': metric_name,
            'V5': v5_val if v5_data else None,
            'V6': v6_val if v6_data else None,
            'V7': v7_val if v7_data else None,
            'Format': fmt
        })
        
        print(f"\n  {metric_name}:")
        if fmt == '%':
            if v5_data:
                print(f"    V5: {v5_val:.1%}")
            if v6_data:
                print(f"    V6: {v6_val:.1%}")
            if v7_data:
                print(f"    V7: {v7_val:.1%}")
        else:
            if v5_data:
                print(f"    V5: {v5_val:.4f}")
            if v6_data:
                print(f"    V6: {v6_val:.4f}")
            if v7_data:
                print(f"    V7: {v7_val:.4f}")
    
    # Profitability metrics
    print(f"\n{'='*80}")
    print("Profitability Metrics:")
    print(f"{'='*80}")
    
    profit_metrics = [
        ('total_return', 'Total Return', '%'),
        ('profit_factor', 'Profit Factor', 'abs'),
        ('sharpe_ratio', 'Sharpe Ratio', 'abs'),
        ('max_drawdown', 'Max Drawdown', '%')
    ]
    
    for metric_key, metric_name, fmt in profit_metrics:
        v5_val = v5_profit.get(metric_key, 0) if v5_data else 0
        v6_val = v6_profit.get(metric_key, 0) if v6_data else 0
        v7_val = v7_profit.get(metric_key, 0) if v7_data else 0
        
        print(f"\n  {metric_name}:")
        if fmt == '%':
            if v5_data:
                print(f"    V5: {v5_val:.2%}")
            if v6_data:
                print(f"    V6: {v6_val:.2%}")
            if v7_data:
                print(f"    V7: {v7_val:.2%}")
        else:
            if v5_data:
                print(f"    V5: {v5_val:.2f}")
            if v6_data:
                print(f"    V6: {v6_val:.2f}")
            if v7_data:
                print(f"    V7: {v7_val:.2f}")
    
    # Save CSV
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        output_dir = os.path.dirname(__file__)
        csv_path = os.path.join(output_dir, 'v5_v6_v7_comparison.csv')
        df_comparison.to_csv(csv_path, index=False)
        print(f"\n{'='*80}")
        print(f"Da luu ket qua so sanh vao: {csv_path}")
    
    # Generate markdown report
    generate_comparison_report(v5_data, v6_data, v7_data, comparison_data)
    
    return comparison_data


def generate_comparison_report(v5_data, v6_data, v7_data, comparison_data):
    """Tao markdown report"""
    output_dir = os.path.dirname(__file__)
    md_path = os.path.join(output_dir, 'V5_V6_V7_COMPARISON.md')
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# So sanh ket qua V5, V6, V7\n\n")
        f.write("## Tong quan\n\n")
        f.write("Script nay so sanh ket qua cua 3 versions:\n")
        f.write("- **V5**: Baseline voi all features\n")
        f.write("- **V6**: Improved threshold evaluation\n")
        f.write("- **V7**: Advanced strategies (expectancy scoring, centered predictions, asymmetric scoring)\n\n")
        
        f.write("## Metrics Comparison\n\n")
        f.write("| Metric | V5 | V6 | V7 |\n")
        f.write("|--------|----|----|----|\n")
        
        for row in comparison_data:
            metric = row['Metric']
            v5_val = row['V5']
            v6_val = row['V6']
            v7_val = row['V7']
            fmt = row['Format']
            
            if fmt == '%':
                v5_str = f"{v5_val:.1%}" if v5_val is not None else "N/A"
                v6_str = f"{v6_val:.1%}" if v6_val is not None else "N/A"
                v7_str = f"{v7_val:.1%}" if v7_val is not None else "N/A"
            else:
                v5_str = f"{v5_val:.4f}" if v5_val is not None else "N/A"
                v6_str = f"{v6_val:.4f}" if v6_val is not None else "N/A"
                v7_str = f"{v7_val:.4f}" if v7_val is not None else "N/A"
            
            f.write(f"| {metric} | {v5_str} | {v6_str} | {v7_str} |\n")
        
        f.write("\n## Ket luan\n\n")
        f.write("Xem chi tiet trong file CSV: `v5_v6_v7_comparison.csv`\n")
    
    print(f"Da tao markdown report: {md_path}")


def main():
    """Main function"""
    comparison = compare_three_versions()
    
    if comparison:
        print(f"\n{'='*80}")
        print("HOAN THANH SO SANH")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

