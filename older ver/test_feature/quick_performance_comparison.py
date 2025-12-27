#!/usr/bin/env python3
"""
Quick Performance Comparison
So sanh nhanh cac feature sets va chon best
Tinh average VIF de danh gia multicollinearity
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v4_multistock'))
sys.path.append('..')

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings("ignore")

import config

class QuickPerformanceComparison:
    def __init__(self, data_dir=None):
        """Khoi tao voi data directory"""
        self.data_dir = data_dir or config.DATA_DIR
        self.comparison_results = []
        
    def load_all_results(self):
        """Load ket qua tu tat ca cac test"""
        results_dir = os.path.join(os.path.dirname(__file__), config.RESULTS_DIR)
        
        # Load VIF ablation results
        vif_ablation_path = os.path.join(results_dir, 'quick_vif_ablation_results.csv')
        vif_results = None
        if os.path.exists(vif_ablation_path):
            vif_results = pd.read_csv(vif_ablation_path)
            print(f"Da load VIF ablation results: {len(vif_results)} configurations")
        
        # Load correlation reduction results
        corr_reduction_path = os.path.join(results_dir, 'quick_correlation_reduction_results.csv')
        corr_results = None
        if os.path.exists(corr_reduction_path):
            corr_results = pd.read_csv(corr_reduction_path)
            print(f"Da load correlation reduction results: {len(corr_results)} configurations")
        
        return vif_results, corr_results
    
    def calculate_avg_vif(self, features):
        """Tinh average VIF cho mot feature set"""
        # Load VIF data
        vif_path = os.path.join(config.RESULTS_DIR, 'quick_vif_analysis.csv')
        if not os.path.exists(vif_path):
            return None
        
        vif_df = pd.read_csv(vif_path)
        vif_dict = dict(zip(vif_df['feature'], vif_df['vif']))
        
        # Tinh average VIF cho features co trong set
        feature_vifs = [vif_dict.get(f, 0) for f in features if f in vif_dict]
        if len(feature_vifs) > 0:
            return np.mean(feature_vifs)
        return None
    
    def compare_configurations(self):
        """So sanh tat ca configurations"""
        print("="*60)
        print("QUICK PERFORMANCE COMPARISON")
        print("="*60)
        
        # Load results
        vif_results, corr_results = self.load_all_results()
        
        if vif_results is None and corr_results is None:
            print("Chua co ket qua de so sanh. Vui long chay cac test truoc.")
            return None
        
        # Combine results
        all_results = []
        
        if vif_results is not None:
            for _, row in vif_results.iterrows():
                # Parse kept features
                if 'removed_features' in row and pd.notna(row['removed_features']):
                    removed = str(row['removed_features']).split(',')
                    # Reconstruct kept features (can load tu baseline)
                    # Tam thoi bo qua, se dung num_features
                    pass
                
                all_results.append({
                    'config_name': row.get('test_type', 'unknown'),
                    'combined_win_rate': row.get('combined_win_rate', 0),
                    'buy_win_rate': row.get('buy_win_rate', 0),
                    'sell_win_rate': row.get('sell_win_rate', 0),
                    'rmse': row.get('rmse', 0),
                    'num_features': row.get('num_features', 0),
                    'avg_vif': row.get('avg_vif', None),
                    'source': 'vif_ablation'
                })
        
        if corr_results is not None:
            for _, row in corr_results.iterrows():
                # Parse kept features
                kept_features = []
                if 'kept_features' in row and pd.notna(row['kept_features']):
                    kept_features = str(row['kept_features']).split(',')
                    avg_vif = self.calculate_avg_vif(kept_features)
                else:
                    avg_vif = None
                
                all_results.append({
                    'config_name': row.get('test_type', 'unknown'),
                    'combined_win_rate': row.get('combined_win_rate', 0),
                    'buy_win_rate': row.get('buy_win_rate', 0),
                    'sell_win_rate': row.get('sell_win_rate', 0),
                    'rmse': row.get('rmse', 0),
                    'num_features': row.get('num_features', 0),
                    'avg_vif': avg_vif,
                    'source': 'correlation_reduction'
                })
        
        comparison_df = pd.DataFrame(all_results)
        
        if len(comparison_df) == 0:
            print("Khong co ket qua de so sanh")
            return None
        
        # Tinh score de rank
        # Score = combined_win_rate - (avg_vif/100) - (num_features/100)
        # Uu tien: win rate cao, VIF thap, so features it
        comparison_df['score'] = (
            comparison_df['combined_win_rate'] * 100 - 
            comparison_df['avg_vif'].fillna(0) - 
            comparison_df['num_features'] * 0.1
        )
        
        # Sort theo score
        comparison_df = comparison_df.sort_values('score', ascending=False)
        
        # Hien thi ket qua
        print("\nSo sanh cac configurations:")
        print(f"{'Config':<25s} {'WR':<8s} {'Buy WR':<8s} {'Sell WR':<8s} {'RMSE':<8s} {'Features':<10s} {'Avg VIF':<10s}")
        print("-" * 90)
        
        for _, row in comparison_df.iterrows():
            avg_vif_str = f"{row['avg_vif']:>9.2f}" if pd.notna(row['avg_vif']) else f"{'N/A':>9s}"
            print(f"{row['config_name']:<25s} "
                  f"{row['combined_win_rate']:>6.1%} "
                  f"{row['buy_win_rate']:>6.1%} "
                  f"{row['sell_win_rate']:>6.1%} "
                  f"{row['rmse']:>7.4f} "
                  f"{row['num_features']:>8d} "
                  f"{avg_vif_str}")
        
        # Tim best configuration
        best = comparison_df.iloc[0]
        
        print("\n" + "="*60)
        print("BEST CONFIGURATION")
        print("="*60)
        print(f"Config: {best['config_name']}")
        print(f"  Combined Win Rate: {best['combined_win_rate']:.1%}")
        print(f"  Buy WR: {best['buy_win_rate']:.1%}, Sell WR: {best['sell_win_rate']:.1%}")
        print(f"  RMSE: {best['rmse']:.4f}")
        print(f"  Number of features: {best['num_features']}")
        if pd.notna(best['avg_vif']):
            print(f"  Average VIF: {best['avg_vif']:.2f}")
        print(f"  Source: {best['source']}")
        
        return comparison_df
    
    def get_best_feature_set(self, comparison_df):
        """Lay best feature set tu ket qua so sanh"""
        if comparison_df is None or len(comparison_df) == 0:
            return None
        
        best = comparison_df.iloc[0]
        best_config_name = best['config_name']
        
        # Load feature set tu source
        if best['source'] == 'vif_ablation':
            # Load tu VIF ablation results
            vif_path = os.path.join(config.RESULTS_DIR, 'quick_vif_ablation_results.csv')
            if os.path.exists(vif_path):
                vif_df = pd.read_csv(vif_path)
                best_row = vif_df[vif_df['test_type'] == best_config_name]
                if len(best_row) > 0:
                    # Reconstruct features (can load baseline features va remove)
                    # Tam thoi tra ve None, se xu ly trong runner
                    pass
        
        elif best['source'] == 'correlation_reduction':
            # Load tu correlation reduction results
            corr_path = os.path.join(config.RESULTS_DIR, 'quick_correlation_reduction_results.csv')
            if os.path.exists(corr_path):
                corr_df = pd.read_csv(corr_path)
                best_row = corr_df[corr_df['test_type'] == best_config_name]
                if len(best_row) > 0 and 'kept_features' in best_row.columns:
                    kept_features_str = best_row.iloc[0]['kept_features']
                    if pd.notna(kept_features_str):
                        return kept_features_str.split(',')
        
        return None
    
    def save_comparison(self, comparison_df):
        """Luu ket qua so sanh"""
        results_dir = os.path.join(os.path.dirname(__file__), config.RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        
        save_path = os.path.join(results_dir, 'quick_comparison_results.csv')
        comparison_df.to_csv(save_path, index=False)
        print(f"\nDa luu ket qua so sanh vao {save_path}")
        
        return save_path


def main():
    """Main workflow"""
    comparer = QuickPerformanceComparison()
    comparison_df = comparer.compare_configurations()
    
    if comparison_df is not None:
        comparer.save_comparison(comparison_df)
        
        # Lay best feature set
        best_features = comparer.get_best_feature_set(comparison_df)
        if best_features:
            print(f"\nBest feature set ({len(best_features)} features):")
            for i, feat in enumerate(best_features, 1):
                print(f"  {i:2d}. {feat}")
    
    print("\nHoan thanh quick performance comparison!")


if __name__ == "__main__":
    main()

