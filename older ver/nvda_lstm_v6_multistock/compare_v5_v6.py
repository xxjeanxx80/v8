#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script so sanh ket qua giua v5 va v6
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Paths
v5_dir = os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v5_multistock')
v6_dir = os.path.dirname(__file__)

v5_artifact = os.path.join(v5_dir, 'nvda_lstm_v5_artifact.pth')
v6_artifact = os.path.join(v6_dir, 'nvda_lstm_v6_artifact.pth')

def load_artifact(artifact_path):
    """Load artifact file"""
    if not os.path.exists(artifact_path):
        return None
    try:
        artifact = torch.load(artifact_path, map_location='cpu')
        return artifact
    except Exception as e:
        print(f"Loi khi load {artifact_path}: {e}")
        return None

def run_v5_and_get_results():
    """Chay v5 va lay ket qua (neu chua co artifact)"""
    v5_script = os.path.join(v5_dir, 'nvda_lstm_v5_multistock.py')
    if not os.path.exists(v5_script):
        return None
    
    # Neu da co artifact, khong can chay lai
    if os.path.exists(v5_artifact):
        return load_artifact(v5_artifact)
    
    print("Chua co v5 artifact. Dang chay v5...")
    print("(Ban co the chay thu cong: python nvda_lstm_v5_multistock.py)")
    return None

def compare_artifacts():
    """So sanh artifacts cua v5 va v6"""
    print("="*80)
    print("SO SANH KET QUA V5 vs V6")
    print("="*80)
    
    # Load artifacts
    v5_data = load_artifact(v5_artifact)
    v6_data = load_artifact(v6_artifact)
    
    if v5_data is None:
        print(f"\nKhong tim thay v5 artifact: {v5_artifact}")
        print("Vui long chay v5 truoc de co ket qua so sanh.")
        return
    
    if v6_data is None:
        print(f"\nKhong tim thay v6 artifact: {v6_artifact}")
        print("Vui long chay v6 truoc de co ket qua so sanh.")
        return
    
    # Extract data
    v5_features = v5_data.get('features', [])
    v6_features = v6_data.get('features', [])
    v5_metrics = v5_data.get('metrics', {})
    v6_metrics = v6_data.get('metrics', {})
    
    print(f"\n{'='*80}")
    print("1. SO LUONG FEATURES")
    print(f"{'='*80}")
    print(f"  v5: {len(v5_features)} features")
    print(f"  v6: {len(v6_features)} features")
    print(f"  Thay doi: {len(v6_features) - len(v5_features):+d} features ({((len(v6_features) - len(v5_features)) / len(v5_features) * 100):+.1f}%)")
    
    print(f"\n{'='*80}")
    print("2. FEATURES COMPARISON")
    print(f"{'='*80}")
    
    v5_set = set(v5_features)
    v6_set = set(v6_features)
    
    removed = v5_set - v6_set
    added = v6_set - v5_set
    common = v5_set & v6_set
    
    print(f"  Features chung: {len(common)}")
    print(f"  Features bi loai bo (v5 co, v6 khong): {len(removed)}")
    if removed:
        print(f"    - {', '.join(sorted(removed))}")
    print(f"  Features duoc them (v6 co, v5 khong): {len(added)}")
    if added:
        print(f"    - {', '.join(sorted(added))}")
    
    print(f"\n{'='*80}")
    print("3. METRICS COMPARISON")
    print(f"{'='*80}")
    
    # So sanh metrics
    metrics_to_compare = [
        ('rmse', 'RMSE', 'lower'),
        ('mae', 'MAE', 'lower'),
        ('buy_wr', 'Buy Win Rate', 'higher'),
        ('sell_wr', 'Sell Win Rate', 'higher'),
        ('combined_wr', 'Combined Win Rate', 'higher'),
    ]
    
    comparison_data = []
    
    for metric_key, metric_name, better in metrics_to_compare:
        v5_val = v5_metrics.get(metric_key, 0)
        v6_val = v6_metrics.get(metric_key, 0)
        
        if v5_val == 0 and v6_val == 0:
            continue
        
        if better == 'lower':
            change_pct = ((v6_val - v5_val) / v5_val * 100) if v5_val != 0 else 0
            better_str = "Tốt hơn" if v6_val < v5_val else "Kém hơn"
        else:
            change_pct = ((v6_val - v5_val) / v5_val * 100) if v5_val != 0 else 0
            better_str = "Tốt hơn" if v6_val > v5_val else "Kém hơn"
        
        comparison_data.append({
            'Metric': metric_name,
            'v5': v5_val,
            'v6': v6_val,
            'Thay doi': f"{change_pct:+.2f}%",
            'Danh gia': better_str if abs(change_pct) > 1 else "Tương đương"
        })
        
        print(f"\n  {metric_name}:")
        if metric_key in ['buy_wr', 'sell_wr', 'combined_wr']:
            print(f"    v5: {v5_val:.2%}")
            print(f"    v6: {v6_val:.2%}")
        else:
            print(f"    v5: {v5_val:.4f}")
            print(f"    v6: {v6_val:.4f}")
        print(f"    Thay doi: {change_pct:+.2f}% ({better_str})")
    
    # Tao DataFrame de hien thi
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(f"\n{'='*80}")
        print("4. BANG SO SANH")
        print(f"{'='*80}")
        print(df_comparison.to_string(index=False))
    
    # Tong ket
    print(f"\n{'='*80}")
    print("5. TONG KET")
    print(f"{'='*80}")
    
    # Dem so metrics tot hon
    better_count = 0
    worse_count = 0
    
    for row in comparison_data:
        if row['Danh gia'] == "Tốt hơn":
            better_count += 1
        elif row['Danh gia'] == "Kém hơn":
            worse_count += 1
    
    print(f"\n  Metrics tot hon v5: {better_count}")
    print(f"  Metrics kem hon v5: {worse_count}")
    print(f"  Metrics tuong duong: {len(comparison_data) - better_count - worse_count}")
    
    # Danh gia chung
    if better_count > worse_count:
        print(f"\n  => v6 TOT HON v5 ve mat metrics")
    elif worse_count > better_count:
        print(f"\n  => v6 KEM HON v5 ve mat metrics")
    else:
        print(f"\n  => v6 TUONG DUONG v5 ve mat metrics")
    
    # Luu ket qua
    if comparison_data:
        output_path = os.path.join(v6_dir, 'v5_v6_comparison.csv')
        df_comparison.to_csv(output_path, index=False)
        print(f"\n  Da luu ket qua so sanh vao: {output_path}")
    
    return comparison_data

def main():
    """Main function"""
    comparison = compare_artifacts()
    
    if comparison:
        print(f"\n{'='*80}")
        print("HOAN THANH SO SANH")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()

