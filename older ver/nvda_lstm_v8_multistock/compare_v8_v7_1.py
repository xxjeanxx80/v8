#!/usr/bin/env python3
"""
So sanh ket qua V8 vs V7.1

Load artifacts tu V8 va V7.1, so sanh metrics:
- Win rates (buy, sell, combined)
- Expectancy
- Profitability (total return, profit factor, Sharpe)
- Model stability
- Training metrics
- Test period differences (V8: 2024-2025, V7.1: 2025 only)
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


def compare_v8_v7_1():
    """So sanh metrics giua V8 va V7.1"""
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths to artifacts
    v8_artifact = os.path.join(workspace_root, 'nvda_lstm_v8_multistock', 'nvda_lstm_v8_artifact.pth')
    v7_1_artifact = os.path.join(workspace_root, 'nvda_lstm_v7_multistock', 'nvda_lstm_v7_1_artifact.pth')
    
    # Load artifacts
    v8_data = load_artifact(v8_artifact)
    v7_1_data = load_artifact(v7_1_artifact)
    
    print(f"\n{'='*80}")
    print("SO SANH KET QUA V8 vs V7.1")
    print(f"{'='*80}")
    
    if v8_data is None:
        print("  V8 artifact khong tim thay. Chay V8 truoc.")
    if v7_1_data is None:
        print("  V7.1 artifact khong tim thay. Chay V7.1 truoc.")
    
    if v8_data is None and v7_1_data is None:
        print("\nKhong co artifact nao de so sanh. Vui long chay cac versions truoc.")
        return None
    
    # Extract metrics
    v8_test_metrics = v8_data.get('test_metrics', {}) if v8_data else {}
    v7_1_test_metrics = v7_1_data.get('test_metrics', {}) if v7_1_data else {}
    
    v8_pretrain = v8_data.get('pretrain_metrics', {}) if v8_data else {}
    v7_1_pretrain = v7_1_data.get('pretrain_metrics', {}) if v7_1_data else {}
    
    v8_finetune = v8_data.get('finetune_metrics', {}) if v8_data else {}
    v7_1_finetune = v7_1_data.get('finetune_metrics', {}) if v7_1_data else {}
    
    v8_profit = v8_data.get('test_profitability', {}) if v8_data else {}
    v7_1_profit = v7_1_data.get('test_profitability', {}) if v7_1_data else {}
    
    # Strategy differences
    print(f"\n{'='*80}")
    print("Strategy Differences:")
    print(f"{'='*80}")
    print(f"\nV7.1 Strategy:")
    print(f"  - Pretrain: 2015-2020 (5 nam)")
    print(f"  - Fine-tune: 2021-2024 (4 nam)")
    print(f"  - Validation: 2023-2024 (NVDA only)")
    print(f"  - Test: 2025 (1 nam) - OUT-OF-SAMPLE")
    
    print(f"\nV8 Strategy:")
    print(f"  - Pretrain: 2015-2020 (5 nam) - GIONG V7.1")
    print(f"  - Fine-tune: 2021-2023 (3 nam) - NGAN HON V7.1")
    print(f"  - Validation: 2022-2023 (NVDA only) - SOM HON V7.1")
    print(f"  - Test: 2024-2025 (2 nam) - DAI HON V7.1, BAO GOM CA 2024")
    
    # Training metrics
    print(f"\n{'='*80}")
    print("Training Metrics:")
    print(f"{'='*80}")
    
    print(f"\nPretrain Best Val Loss:")
    if v8_pretrain:
        v8_pretrain_loss = v8_pretrain.get('best_val_loss', None)
        if v8_pretrain_loss is not None:
            print(f"  V8: {v8_pretrain_loss:.6f}")
        else:
            print(f"  V8: N/A")
    if v7_1_pretrain:
        v7_1_pretrain_loss = v7_1_pretrain.get('best_val_loss', None)
        if v7_1_pretrain_loss is not None:
            print(f"  V7.1: {v7_1_pretrain_loss:.6f}")
        else:
            print(f"  V7.1: N/A")
    
    print(f"\nFine-tune Best Val Loss:")
    if v8_finetune:
        v8_finetune_loss = v8_finetune.get('best_val_loss', None)
        if v8_finetune_loss is not None:
            print(f"  V8: {v8_finetune_loss:.6f}")
        else:
            print(f"  V8: N/A")
    if v7_1_finetune:
        v7_1_finetune_loss = v7_1_finetune.get('best_val_loss', None)
        if v7_1_finetune_loss is not None:
            print(f"  V7.1: {v7_1_finetune_loss:.6f}")
        else:
            print(f"  V7.1: N/A")
    
    # Test metrics
    print(f"\n{'='*80}")
    print("Test Metrics (V8: 2024-2025, V7.1: 2025 only):")
    print(f"{'='*80}")
    
    metrics_to_compare = [
        ('buy_wr', 'Buy Win Rate', '%'),
        ('sell_wr', 'Sell Win Rate', '%'),
        ('combined_wr', 'Combined Win Rate', '%'),
        ('buy_expectancy', 'Buy Expectancy', 'abs'),
        ('sell_expectancy', 'Sell Expectancy', 'abs'),
        ('buy_coverage', 'Buy Coverage', '%'),
        ('sell_coverage', 'Sell Coverage', '%'),
        ('coverage', 'Total Coverage', '%'),
        ('rmse', 'RMSE', 'abs'),
        ('mae', 'MAE', 'abs')
    ]
    
    comparison_data = []
    
    for metric_key, metric_name, fmt in metrics_to_compare:
        v8_val = v8_test_metrics.get(metric_key, 0) if v8_data else 0
        v7_1_val = v7_1_test_metrics.get(metric_key, 0) if v7_1_data else 0
        
        comparison_data.append({
            'Metric': metric_name,
            'V8': v8_val if v8_data else None,
            'V7.1': v7_1_val if v7_1_data else None,
            'Format': fmt
        })
        
        print(f"\n  {metric_name}:")
        if fmt == '%':
            if v8_data:
                print(f"    V8: {v8_val:.1%}")
            if v7_1_data:
                print(f"    V7.1: {v7_1_val:.1%}")
            if v8_data and v7_1_data:
                diff = v8_val - v7_1_val
                print(f"    Chenh lech: {diff:+.1%} ({'V8 tot hon' if diff > 0 else 'V7.1 tot hon' if diff < 0 else 'Bang nhau'})")
        else:
            if v8_data:
                print(f"    V8: {v8_val:.4f}")
            if v7_1_data:
                print(f"    V7.1: {v7_1_val:.4f}")
            if v8_data and v7_1_data:
                diff = v8_val - v7_1_val
                print(f"    Chenh lech: {diff:+.4f} ({'V8 tot hon' if diff < 0 and 'rmse' in metric_key.lower() or 'mae' in metric_key.lower() else 'V8 tot hon' if diff > 0 else 'V7.1 tot hon' if diff < 0 else 'Bang nhau'})")
    
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
        v8_val = v8_profit.get(metric_key, 0) if v8_data else 0
        v7_1_val = v7_1_profit.get(metric_key, 0) if v7_1_data else 0
        
        print(f"\n  {metric_name}:")
        if fmt == '%':
            if v8_data:
                print(f"    V8: {v8_val:.2%}")
            if v7_1_data:
                print(f"    V7.1: {v7_1_val:.2%}")
            if v8_data and v7_1_data:
                diff = v8_val - v7_1_val
                print(f"    Chenh lech: {diff:+.2%} ({'V8 tot hon' if diff > 0 else 'V7.1 tot hon' if diff < 0 else 'Bang nhau'})")
        else:
            if v8_data:
                print(f"    V8: {v8_val:.2f}")
            if v7_1_data:
                print(f"    V7.1: {v7_1_val:.2f}")
            if v8_data and v7_1_data:
                diff = v8_val - v7_1_val
                if metric_key == 'max_drawdown':
                    print(f"    Chenh lech: {diff:+.2%} ({'V8 tot hon' if diff > 0 else 'V7.1 tot hon' if diff < 0 else 'Bang nhau'})")
                else:
                    print(f"    Chenh lech: {diff:+.2f} ({'V8 tot hon' if diff > 0 else 'V7.1 tot hon' if diff < 0 else 'Bang nhau'})")
    
    # V8: Metrics by year (2024 vs 2025)
    if v8_data and v8_data.get('test_by_year'):
        print(f"\n{'='*80}")
        print("V8 Metrics by Year (2024 vs 2025):")
        print(f"{'='*80}")
        
        v8_by_year = v8_data.get('test_by_year', {})
        if '2024' in v8_by_year and '2025' in v8_by_year:
            print(f"\n  2024:")
            print(f"    Buy Win Rate: {v8_by_year['2024'].get('buy_wr', 0):.1%}")
            print(f"    Sell Win Rate: {v8_by_year['2024'].get('sell_wr', 0):.1%}")
            print(f"    Combined Win Rate: {v8_by_year['2024'].get('combined_wr', 0):.1%}")
            print(f"    Buy Coverage: {v8_by_year['2024'].get('buy_coverage', 0):.1%}")
            print(f"    Sell Coverage: {v8_by_year['2024'].get('sell_coverage', 0):.1%}")
            
            print(f"\n  2025:")
            print(f"    Buy Win Rate: {v8_by_year['2025'].get('buy_wr', 0):.1%}")
            print(f"    Sell Win Rate: {v8_by_year['2025'].get('sell_wr', 0):.1%}")
            print(f"    Combined Win Rate: {v8_by_year['2025'].get('combined_wr', 0):.1%}")
            print(f"    Buy Coverage: {v8_by_year['2025'].get('buy_coverage', 0):.1%}")
            print(f"    Sell Coverage: {v8_by_year['2025'].get('sell_coverage', 0):.1%}")
            
            # So sanh 2024 vs 2025 trong V8
            print(f"\n  So sanh 2024 vs 2025 trong V8:")
            buy_wr_2024 = v8_by_year['2024'].get('buy_wr', 0)
            buy_wr_2025 = v8_by_year['2025'].get('buy_wr', 0)
            combined_wr_2024 = v8_by_year['2024'].get('combined_wr', 0)
            combined_wr_2025 = v8_by_year['2025'].get('combined_wr', 0)
            
            print(f"    Buy Win Rate: 2024 ({buy_wr_2024:.1%}) vs 2025 ({buy_wr_2025:.1%}) - Chenh lech: {buy_wr_2024 - buy_wr_2025:+.1%}")
            print(f"    Combined Win Rate: 2024 ({combined_wr_2024:.1%}) vs 2025 ({combined_wr_2025:.1%}) - Chenh lech: {combined_wr_2024 - combined_wr_2025:+.1%}")
            
            # SO SANH CONG BANG: V8 (2025 only) vs V7.1 (2025 only)
            print(f"\n{'='*80}")
            print("SO SANH CONG BANG: V8 (2025 only) vs V7.1 (2025 only):")
            print(f"{'='*80}")
            
            v8_2025_buy_wr = v8_by_year['2025'].get('buy_wr', 0)
            v8_2025_sell_wr = v8_by_year['2025'].get('sell_wr', 0)
            v8_2025_combined_wr = v8_by_year['2025'].get('combined_wr', 0)
            v7_1_buy_wr = v7_1_test_metrics.get('buy_wr', 0) if v7_1_data else 0
            v7_1_sell_wr = v7_1_test_metrics.get('sell_wr', 0) if v7_1_data else 0
            v7_1_combined_wr = v7_1_test_metrics.get('combined_wr', 0) if v7_1_data else 0
            
            # Kết quả mới từ terminal (nếu có)
            # V7.1 mới: Buy Win Rate: 63.6%, Sell Win Rate: 19.0%, Combined: 41.3%
            v7_1_new_buy_wr = 0.636  # Từ terminal output
            v7_1_new_sell_wr = 0.190  # Từ terminal output
            v7_1_new_combined_wr = 0.413  # Từ terminal output
            
            print(f"\n  Buy Win Rate (2025 only):")
            print(f"    V8: {v8_2025_buy_wr:.1%}")
            print(f"    V7.1 (artifact): {v7_1_buy_wr:.1%}")
            print(f"    V7.1 (new run): {v7_1_new_buy_wr:.1%}  <-- Kết quả mới từ terminal")
            if v8_data:
                diff_artifact = v8_2025_buy_wr - v7_1_buy_wr
                diff_new = v8_2025_buy_wr - v7_1_new_buy_wr
                print(f"    Chenh lech (vs artifact): {diff_artifact:+.1%}")
                print(f"    Chenh lech (vs new run): {diff_new:+.1%} ({'V8 tot hon' if diff_new > 0 else 'V7.1 tot hon' if diff_new < 0 else 'Bang nhau'})")
            
            print(f"\n  Sell Win Rate (2025 only):")
            print(f"    V8: {v8_2025_sell_wr:.1%}")
            print(f"    V7.1 (artifact): {v7_1_sell_wr:.1%}")
            print(f"    V7.1 (new run): {v7_1_new_sell_wr:.1%}  <-- Kết quả mới từ terminal")
            if v8_data:
                diff_artifact = v8_2025_sell_wr - v7_1_sell_wr
                diff_new = v8_2025_sell_wr - v7_1_new_sell_wr
                print(f"    Chenh lech (vs artifact): {diff_artifact:+.1%}")
                print(f"    Chenh lech (vs new run): {diff_new:+.1%} ({'V8 tot hon' if diff_new > 0 else 'V7.1 tot hon' if diff_new < 0 else 'Bang nhau'})")
            
            print(f"\n  Combined Win Rate (2025 only):")
            print(f"    V8: {v8_2025_combined_wr:.1%}")
            print(f"    V7.1 (artifact): {v7_1_combined_wr:.1%}")
            print(f"    V7.1 (new run): {v7_1_new_combined_wr:.1%}  <-- Kết quả mới từ terminal")
            if v8_data:
                diff_artifact = v8_2025_combined_wr - v7_1_combined_wr
                diff_new = v8_2025_combined_wr - v7_1_new_combined_wr
                print(f"    Chenh lech (vs artifact): {diff_artifact:+.1%}")
                print(f"    Chenh lech (vs new run): {diff_new:+.1%} ({'V8 tot hon' if diff_new > 0 else 'V7.1 tot hon' if diff_new < 0 else 'Bang nhau'})")
    
    # Save CSV
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        output_dir = os.path.dirname(__file__)
        csv_path = os.path.join(output_dir, 'v8_v7_1_comparison.csv')
        df_comparison.to_csv(csv_path, index=False)
        print(f"\n{'='*80}")
        print(f"Da luu ket qua so sanh vao: {csv_path}")
    
    # Generate markdown report
    generate_comparison_report(v8_data, v7_1_data, comparison_data)
    
    return comparison_data


def generate_comparison_report(v8_data, v7_1_data, comparison_data):
    """Tao markdown report"""
    output_dir = os.path.dirname(__file__)
    md_path = os.path.join(output_dir, 'V8_V7_1_COMPARISON.md')
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# So sanh ket qua V8 vs V7.1\n\n")
        f.write("## Tong quan\n\n")
        f.write("Script nay so sanh ket qua cua 2 versions:\n")
        f.write("- **V7.1**: No-leakage strategy (pretrain 2015-2020, fine-tune 2021-2024, test 2025)\n")
        f.write("- **V8**: Extended test period (pretrain 2015-2020, fine-tune 2021-2023, test 2024-2025)\n\n")
        
        f.write("## Strategy Differences\n\n")
        f.write("| Aspect | V7.1 | V8 |\n")
        f.write("|--------|------|----|\n")
        f.write("| Pretrain | 2015-2020 (5 nam) | 2015-2020 (5 nam) - GIONG |\n")
        f.write("| Fine-tune | 2021-2024 (4 nam) | 2021-2023 (3 nam) - NGAN HON |\n")
        f.write("| Validation | 2023-2024 | 2022-2023 - SOM HON |\n")
        f.write("| Test | 2025 (1 nam) | 2024-2025 (2 nam) - DAI HON |\n\n")
        
        f.write("## Metrics Comparison\n\n")
        f.write("| Metric | V8 | V7.1 | Difference |\n")
        f.write("|--------|----|----|------------|\n")
        
        for row in comparison_data:
            metric = row['Metric']
            v8_val = row['V8']
            v7_1_val = row['V7.1']
            fmt = row['Format']
            
            if fmt == '%':
                v8_str = f"{v8_val:.1%}" if v8_val is not None else "N/A"
                v7_1_str = f"{v7_1_val:.1%}" if v7_1_val is not None else "N/A"
                if v8_val is not None and v7_1_val is not None:
                    diff = v8_val - v7_1_val
                    diff_str = f"{diff:+.1%}"
                else:
                    diff_str = "N/A"
            else:
                v8_str = f"{v8_val:.4f}" if v8_val is not None else "N/A"
                v7_1_str = f"{v7_1_val:.4f}" if v7_1_val is not None else "N/A"
                if v8_val is not None and v7_1_val is not None:
                    diff = v8_val - v7_1_val
                    diff_str = f"{diff:+.4f}"
                else:
                    diff_str = "N/A"
            
            f.write(f"| {metric} | {v8_str} | {v7_1_str} | {diff_str} |\n")
        
        # V8 Metrics by Year
        if v8_data and v8_data.get('test_by_year'):
            f.write("\n## V8 Metrics by Year (2024 vs 2025)\n\n")
            v8_by_year = v8_data.get('test_by_year', {})
            if '2024' in v8_by_year and '2025' in v8_by_year:
                f.write("| Metric | 2024 | 2025 |\n")
                f.write("|--------|------|------|\n")
                f.write(f"| Buy Win Rate | {v8_by_year['2024'].get('buy_wr', 0):.1%} | {v8_by_year['2025'].get('buy_wr', 0):.1%} |\n")
                f.write(f"| Sell Win Rate | {v8_by_year['2024'].get('sell_wr', 0):.1%} | {v8_by_year['2025'].get('sell_wr', 0):.1%} |\n")
                f.write(f"| Combined Win Rate | {v8_by_year['2024'].get('combined_wr', 0):.1%} | {v8_by_year['2025'].get('combined_wr', 0):.1%} |\n")
                f.write(f"| Buy Coverage | {v8_by_year['2024'].get('buy_coverage', 0):.1%} | {v8_by_year['2025'].get('buy_coverage', 0):.1%} |\n")
                f.write(f"| Sell Coverage | {v8_by_year['2024'].get('sell_coverage', 0):.1%} | {v8_by_year['2025'].get('sell_coverage', 0):.1%} |\n")
        
        f.write("\n## Ket luan\n\n")
        f.write("Xem chi tiet trong file CSV: `v8_v7_1_comparison.csv`\n")
        f.write("\n### Key Insights:\n")
        f.write("- V8 co test period dai hon (2024-2025) so voi V7.1 (chi 2025)\n")
        f.write("- V8 fine-tune ngan hon (2021-2023) so voi V7.1 (2021-2024)\n")
        f.write("- V8 validation som hon (2022-2023) so voi V7.1 (2023-2024)\n")
        f.write("- V8 cho phep danh gia model tren 2 nam thay vi 1 nam\n")
    
    print(f"Da tao markdown report: {md_path}")


def main():
    """Main function"""
    comparison = compare_v8_v7_1()
    
    if comparison:
        print(f"\n{'='*80}")
        print("HOAN THANH SO SANH V8 vs V7.1")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

