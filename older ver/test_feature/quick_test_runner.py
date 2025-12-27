#!/usr/bin/env python3
"""
Quick Test Runner
Chay tat ca tests tuan tu va tao best config cho v5
"""

import sys
import os
import pandas as pd
import time

# Import cac test modules
from quick_multicollinearity_test import QuickMulticollinearityTester
from quick_vif_ablation import QuickVIFAblation
from quick_correlation_reduction import QuickCorrelationReduction
from quick_performance_comparison import QuickPerformanceComparison

import config

class QuickTestRunner:
    def __init__(self):
        """Khoi tao runner"""
        self.start_time = None
        self.results_dir = os.path.join(os.path.dirname(__file__), config.RESULTS_DIR)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_phase1_multicollinearity(self):
        """Phase 1: Quick Multicollinearity Analysis"""
        print("\n" + "="*80)
        print("PHASE 1: QUICK MULTICOLLINEARITY ANALYSIS")
        print("="*80)
        
        tester = QuickMulticollinearityTester()
        vif_df, corr_results, suggestions = tester.run_analysis()
        
        print("\nPhase 1 hoan thanh!")
        return True
    
    def run_phase2_vif_ablation(self):
        """Phase 2: VIF-based Ablation"""
        print("\n" + "="*80)
        print("PHASE 2: VIF-BASED ABLATION TEST")
        print("="*80)
        
        tester = QuickVIFAblation()
        results = tester.run_ablation()
        tester.save_results(results)
        
        print("\nPhase 2 hoan thanh!")
        return True
    
    def run_phase3_correlation_reduction(self):
        """Phase 3: Correlation-based Reduction"""
        print("\n" + "="*80)
        print("PHASE 3: CORRELATION-BASED REDUCTION")
        print("="*80)
        
        tester = QuickCorrelationReduction()
        results = tester.run_reduction_tests()
        tester.save_results(results)
        
        print("\nPhase 3 hoan thanh!")
        return True
    
    def run_phase4_comparison(self):
        """Phase 4: Performance Comparison"""
        print("\n" + "="*80)
        print("PHASE 4: PERFORMANCE COMPARISON")
        print("="*80)
        
        comparer = QuickPerformanceComparison()
        comparison_df = comparer.compare_configurations()
        
        if comparison_df is not None:
            comparer.save_comparison(comparison_df)
            print("\nPhase 4 hoan thanh!")
            return comparison_df
        else:
            print("\nPhase 4: Khong co ket qua de so sanh")
            return None
    
    def extract_best_feature_set(self, comparison_df):
        """Trich xuat best feature set tu ket qua so sanh"""
        if comparison_df is None or len(comparison_df) == 0:
            return None
        
        best = comparison_df.iloc[0]
        best_config_name = best['config_name']
        best_source = best['source']
        
        # Load v5 baseline features
        v5_path = os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v5_multistock', 'nvda_lstm_v5_multistock.py')
        sys.path.insert(0, os.path.dirname(v5_path))
        try:
            from nvda_lstm_v5_multistock import load_optimized_features
            baseline_features = load_optimized_features()
        except:
            # Fallback
            baseline_features = [
                'rsi14','macd','macd_bullish','macd_signal','macd_hist',
                'atr','bb_bandwidth','bb_percent',
                'volume_ratio','obv','volume_sma20',
                'daily_return','price_change','return_3d','return_5d','return_10d','return_20d',
                'hl_spread_pct','oc_spread','oc_spread_pct',
                'bb_squeeze','rsi_overbought','rsi_oversold',
                'sox_beta','sox_correlation'
            ]
        
        best_features = None
        
        if best_source == 'vif_ablation':
            # Load tu VIF ablation results
            vif_path = os.path.join(self.results_dir, 'quick_vif_ablation_results.csv')
            if os.path.exists(vif_path):
                vif_df = pd.read_csv(vif_path)
                best_row = vif_df[vif_df['test_type'] == best_config_name]
                
                if len(best_row) > 0:
                    removed_str = best_row.iloc[0].get('removed_features', '')
                    if pd.notna(removed_str) and removed_str:
                        removed = str(removed_str).split(',')
                        best_features = [f for f in baseline_features if f not in removed]
                    else:
                        # Baseline
                        best_features = baseline_features.copy()
        
        elif best_source == 'correlation_reduction':
            # Load tu correlation reduction results
            corr_path = os.path.join(self.results_dir, 'quick_correlation_reduction_results.csv')
            if os.path.exists(corr_path):
                corr_df = pd.read_csv(corr_path)
                best_row = corr_df[corr_df['test_type'] == best_config_name]
                
                if len(best_row) > 0:
                    kept_str = best_row.iloc[0].get('kept_features', '')
                    if pd.notna(kept_str) and kept_str:
                        best_features = str(kept_str).split(',')
                    else:
                        best_features = baseline_features.copy()
        
        # Fallback: neu khong tim thay, dung baseline
        if best_features is None:
            best_features = baseline_features.copy()
        
        return best_features
    
    def create_best_config(self, best_features):
        """Tao file config cho best feature set"""
        if best_features is None or len(best_features) == 0:
            print("Khong the tao config: khong co best features")
            return None
        
        # Tao rationale
        rationale = [
            f"Best feature set tu quick test ({len(best_features)} features)",
            "Giam multicollinearity bang cach loai bo features co VIF cao va redundant",
            "Dua tren ket qua VIF ablation va correlation reduction tests"
        ]
        
        # Tao config DataFrame
        config_data = {
            'features': [str(best_features)],
            'count': [len(best_features)],
            'rationale': ['; '.join(rationale)]
        }
        
        config_df = pd.DataFrame(config_data)
        
        # Luu vao results directory
        save_path = os.path.join(self.results_dir, 'best_feature_set_v5.csv')
        config_df.to_csv(save_path, index=False)
        print(f"\nDa luu best feature set vao {save_path}")
        
        # Cap nhat optimized_feature_config.csv neu can
        update_optimized = input("\nBan co muon cap nhat optimized_feature_config.csv? (y/n): ").strip().lower()
        if update_optimized == 'y':
            optimized_path = os.path.join(os.path.dirname(__file__), 'optimized_feature_config.csv')
            config_df.to_csv(optimized_path, index=False)
            print(f"Da cap nhat {optimized_path}")
        
        return save_path
    
    def run_all(self):
        """Chay tat ca cac phases"""
        self.start_time = time.time()
        
        print("="*80)
        print("QUICK TEST RUNNER - Giam Multicollinearity va Tang Do Chinh Xac")
        print("="*80)
        print(f"Thoi gian uoc tinh: ~30 phut")
        print(f"Device: {config.DEVICE}")
        print("="*80)
        
        try:
            # Phase 1: Multicollinearity Analysis
            phase1_start = time.time()
            if not self.run_phase1_multicollinearity():
                print("Phase 1 that bai!")
                return False
            phase1_time = time.time() - phase1_start
            print(f"\nPhase 1 mat {phase1_time/60:.1f} phut")
            
            # Phase 2: VIF Ablation
            phase2_start = time.time()
            if not self.run_phase2_vif_ablation():
                print("Phase 2 that bai!")
                return False
            phase2_time = time.time() - phase2_start
            print(f"\nPhase 2 mat {phase2_time/60:.1f} phut")
            
            # Phase 3: Correlation Reduction
            phase3_start = time.time()
            if not self.run_phase3_correlation_reduction():
                print("Phase 3 that bai!")
                return False
            phase3_time = time.time() - phase3_start
            print(f"\nPhase 3 mat {phase3_time/60:.1f} phut")
            
            # Phase 4: Comparison
            phase4_start = time.time()
            comparison_df = self.run_phase4_comparison()
            phase4_time = time.time() - phase4_start
            print(f"\nPhase 4 mat {phase4_time/60:.1f} phut")
            
            # Phase 5: Create Best Config
            print("\n" + "="*80)
            print("PHASE 5: TAO BEST CONFIG CHO V5")
            print("="*80)
            
            best_features = self.extract_best_feature_set(comparison_df)
            
            if best_features:
                print(f"\nBest feature set ({len(best_features)} features):")
                for i, feat in enumerate(best_features, 1):
                    print(f"  {i:2d}. {feat}")
                
                self.create_best_config(best_features)
            else:
                print("Khong the xac dinh best feature set")
            
            # Tong ket
            total_time = time.time() - self.start_time
            print("\n" + "="*80)
            print("TONG KET")
            print("="*80)
            print(f"Tong thoi gian: {total_time/60:.1f} phut")
            print(f"  Phase 1: {phase1_time/60:.1f} phut")
            print(f"  Phase 2: {phase2_time/60:.1f} phut")
            print(f"  Phase 3: {phase3_time/60:.1f} phut")
            print(f"  Phase 4: {phase4_time/60:.1f} phut")
            print("\nHoan thanh tat ca tests!")
            
            return True
            
        except Exception as e:
            print(f"\nLoi khi chay tests: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    runner = QuickTestRunner()
    success = runner.run_all()
    
    if success:
        print("\nTat ca tests da hoan thanh thanh cong!")
        print("Kiem tra thu muc results/ de xem ket qua chi tiet.")
    else:
        print("\nCo loi xay ra trong qua trinh test.")


if __name__ == "__main__":
    main()

