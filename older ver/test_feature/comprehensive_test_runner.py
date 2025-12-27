#!/usr/bin/env python3
"""
Comprehensive Test Runner
Chay tat ca cac test va tong hop ket qua
So sanh cac phuong an va chon best configuration
Generate final report
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v4_multistock'))
sys.path.append('..')

import warnings
warnings.filterwarnings("ignore")

import config
from threshold_optimization import ThresholdOptimizer
from feature_combination_test import FeatureCombinationTester
from advanced_ablation_test import AdvancedAblationTester
from trading_strategy_evaluator import TradingStrategyEvaluator

class ComprehensiveTestRunner:
    def __init__(self):
        """Khoi tao runner"""
        self.results_summary = {}
        self.best_configuration = {}
        self.recommendations = []
        
    def run_threshold_optimization(self):
        """Chay threshold optimization test"""
        print("\n" + "="*60)
        print("PHASE 1: THRESHOLD OPTIMIZATION")
        print("="*60)
        
        optimizer = ThresholdOptimizer()
        results, best_combined, best_weighted, best_sharpe = optimizer.run_optimization()
        
        optimizer.visualize_results()
        optimizer.save_results()
        
        self.results_summary['threshold_optimization'] = {
            'best_combined': best_combined.to_dict(),
            'best_weighted': best_weighted.to_dict(),
            'best_sharpe': best_sharpe.to_dict()
        }
        
        return best_combined, best_weighted, best_sharpe
    
    def run_feature_combination(self, buy_threshold_pct=75, sell_threshold_pct=75):
        """Chay feature combination test"""
        print("\n" + "="*60)
        print("PHASE 2: FEATURE COMBINATION TESTING")
        print("="*60)
        
        tester = FeatureCombinationTester()
        results = tester.run_comprehensive_test(buy_threshold_pct, sell_threshold_pct)
        
        tester.save_results()
        
        if len(results) > 0:
            best = results.loc[results['combined_win_rate'].idxmax()]
            self.results_summary['feature_combination'] = {
                'best_config': best.to_dict(),
                'total_tests': len(results)
            }
            return best
        return None
    
    def run_advanced_ablation(self, buy_threshold_pct=75, sell_threshold_pct=75):
        """Chay advanced ablation test"""
        print("\n" + "="*60)
        print("PHASE 3: ADVANCED ABLATION TESTING")
        print("="*60)
        
        tester = AdvancedAblationTester()
        results = tester.run_comprehensive_test(buy_threshold_pct, sell_threshold_pct)
        
        tester.save_results()
        
        if len(results) > 0:
            best = results.loc[results['combined_win_rate'].idxmax()]
            self.results_summary['advanced_ablation'] = {
                'best_config': best.to_dict(),
                'total_tests': len(results)
            }
            return best
        return None
    
    def run_trading_evaluation(self, feature_indices, feature_names,
                               buy_threshold_pct=75, sell_threshold_pct=75):
        """Chay trading strategy evaluation"""
        print("\n" + "="*60)
        print("PHASE 4: TRADING STRATEGY EVALUATION")
        print("="*60)
        
        evaluator = TradingStrategyEvaluator()
        metrics, wf_results = evaluator.evaluate_strategy(
            feature_indices, feature_names,
            buy_threshold_pct, sell_threshold_pct
        )
        
        evaluator.save_report()
        
        self.results_summary['trading_evaluation'] = {
            'main_metrics': metrics,
            'walk_forward': wf_results
        }
        
        return metrics
    
    def determine_best_configuration(self):
        """Xac dinh best configuration tu tat ca ket qua"""
        print("\n" + "="*60)
        print("XAC DINH BEST CONFIGURATION")
        print("="*60)
        
        best_config = {
            'buy_threshold_pct': 75,
            'sell_threshold_pct': 75,
            'features': [],
            'metrics': {}
        }
        
        # Lay best threshold tu threshold optimization
        if 'threshold_optimization' in self.results_summary:
            best_threshold = self.results_summary['threshold_optimization']['best_combined']
            best_config['buy_threshold_pct'] = best_threshold['buy_percentile']
            best_config['sell_threshold_pct'] = best_threshold['sell_percentile']
        
        # Lay best features tu feature combination hoac advanced ablation
        best_features = None
        best_wr = 0
        
        if 'feature_combination' in self.results_summary:
            fc_best = self.results_summary['feature_combination']['best_config']
            if fc_best.get('combined_win_rate', 0) > best_wr:
                best_wr = fc_best.get('combined_win_rate', 0)
                if 'test_features' in fc_best:
                    best_features = fc_best['test_features'].split(',')
        
        if 'advanced_ablation' in self.results_summary:
            aa_best = self.results_summary['advanced_ablation']['best_config']
            if aa_best.get('combined_win_rate', 0) > best_wr:
                best_wr = aa_best.get('combined_win_rate', 0)
                if 'test_features' in aa_best:
                    best_features = aa_best['test_features'].split(',')
        
        if best_features:
            best_config['features'] = best_features
        else:
            # Fallback: su dung no_trend features
            from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete
            predictor = NVDA_MultiStock_Complete()
            df, feature_cols = predictor.load_multi_stock_data(config.DATA_DIR)
            no_trend = [f for f in feature_cols if not any(x in f.lower() for x in ['sma', 'price_vs'])]
            best_config['features'] = no_trend
        
        # Lay metrics tu trading evaluation neu co
        if 'trading_evaluation' in self.results_summary:
            best_config['metrics'] = self.results_summary['trading_evaluation']['main_metrics']
        
        self.best_configuration = best_config
        
        print(f"\nBest Configuration:")
        print(f"  Buy Threshold: {best_config['buy_threshold_pct']}%")
        print(f"  Sell Threshold: {best_config['sell_threshold_pct']}%")
        print(f"  Number of Features: {len(best_config['features'])}")
        print(f"  Combined Win Rate: {best_wr:.1%}")
        
        return best_config
    
    def generate_recommendations(self):
        """Tao khuyen nghi implementation"""
        print("\n" + "="*60)
        print("TAO KHUYEN NGHI")
        print("="*60)
        
        recommendations = []
        
        # Kiem tra metrics
        if 'trading_evaluation' in self.results_summary:
            metrics = self.results_summary['trading_evaluation']['main_metrics']
            
            # Kiem tra win rates
            if metrics.get('combined_win_rate', 0) >= config.MIN_COMBINED_WIN_RATE:
                recommendations.append(f"Combined win rate ({metrics['combined_win_rate']:.1%}) dat yeu cau toi thieu ({config.MIN_COMBINED_WIN_RATE:.0%})")
            else:
                recommendations.append(f"Canh bao: Combined win rate ({metrics['combined_win_rate']:.1%}) thap hon yeu cau ({config.MIN_COMBINED_WIN_RATE:.0%})")
            
            if metrics.get('buy_win_rate', 0) >= config.MIN_BUY_WIN_RATE:
                recommendations.append(f"Buy win rate ({metrics['buy_win_rate']:.1%}) dat yeu cau ({config.MIN_BUY_WIN_RATE:.0%})")
            else:
                recommendations.append(f"Canh bao: Buy win rate ({metrics['buy_win_rate']:.1%}) thap hon yeu cau ({config.MIN_BUY_WIN_RATE:.0%})")
            
            if metrics.get('sell_win_rate', 0) >= config.MIN_SELL_WIN_RATE:
                recommendations.append(f"Sell win rate ({metrics['sell_win_rate']:.1%}) dat yeu cau ({config.MIN_SELL_WIN_RATE:.0%})")
            else:
                recommendations.append(f"Canh bao: Sell win rate ({metrics['sell_win_rate']:.1%}) thap hon yeu cau ({config.MIN_SELL_WIN_RATE:.0%})")
            
            # Kiem tra returns
            if metrics.get('total_return', 0) > 0:
                recommendations.append(f"Total return duong ({metrics['total_return']:.4f}), chien luoc co hieu qua")
            else:
                recommendations.append(f"Canh bao: Total return am ({metrics['total_return']:.4f}), can xem xet lai")
            
            # Kiem tra Sharpe ratio
            if metrics.get('sharpe_ratio', 0) > 1.0:
                recommendations.append(f"Sharpe ratio tot ({metrics['sharpe_ratio']:.2f}), risk-adjusted return cao")
            elif metrics.get('sharpe_ratio', 0) > 0:
                recommendations.append(f"Sharpe ratio trung binh ({metrics['sharpe_ratio']:.2f})")
            else:
                recommendations.append(f"Canh bao: Sharpe ratio am ({metrics['sharpe_ratio']:.2f}), can cai thien")
        
        # Kiem tra feature count
        if self.best_configuration:
            num_features = len(self.best_configuration.get('features', []))
            if num_features < 20:
                recommendations.append(f"So luong feature ({num_features}) hop ly, de quan ly")
            elif num_features > 35:
                recommendations.append(f"Canh bao: So luong feature ({num_features}) nhieu, co the giam bot")
        
        # Kiem tra threshold
        if self.best_configuration:
            buy_pct = self.best_configuration.get('buy_threshold_pct', 75)
            sell_pct = self.best_configuration.get('sell_threshold_pct', 75)
            if buy_pct != sell_pct:
                recommendations.append(f"Threshold khac nhau cho buy ({buy_pct}%) va sell ({sell_pct}%) la hop ly")
            else:
                recommendations.append(f"Su dung cung threshold ({buy_pct}%) cho ca buy va sell")
        
        # Khuyen nghi implementation
        recommendations.append("\nKHUYEN NGHI IMPLEMENTATION:")
        recommendations.append("1. Su dung best configuration duoc xac dinh trong best_configuration.json")
        recommendations.append("2. Test lai tren data moi de dam bao tinh on dinh")
        recommendations.append("3. Monitor performance trong production va dieu chinh neu can")
        recommendations.append("4. Su dung walk-forward analysis de cap nhat model dinh ky")
        
        self.recommendations = recommendations
        
        for rec in recommendations:
            print(f"  - {rec}")
        
        return recommendations
    
    def generate_report(self):
        """Tao bao cao tong hop"""
        print("\n" + "="*60)
        print("TAO BAO CAO TONG HOP")
        print("="*60)
        
        report_path = config.OUTPUT_FILES['comprehensive_report']
        
        report_lines = []
        report_lines.append("# BAO CAO TONG HOP TEST FEATURE")
        report_lines.append(f"\nNgay tao: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("\n" + "="*60)
        
        # Tom tat ket qua
        report_lines.append("\n## TOM TAT KET QUA")
        report_lines.append("\n### Threshold Optimization")
        if 'threshold_optimization' in self.results_summary:
            best = self.results_summary['threshold_optimization']['best_combined']
            report_lines.append(f"- Best Combined Win Rate: {best['combined_win_rate']:.1%}")
            report_lines.append(f"- Buy Threshold: {best['buy_percentile']:.0f}%")
            report_lines.append(f"- Sell Threshold: {best['sell_percentile']:.0f}%")
        
        report_lines.append("\n### Feature Combination")
        if 'feature_combination' in self.results_summary:
            best = self.results_summary['feature_combination']['best_config']
            report_lines.append(f"- Best Combined Win Rate: {best.get('combined_win_rate', 0):.1%}")
            report_lines.append(f"- Number of Features: {best.get('num_features', 0)}")
        
        report_lines.append("\n### Advanced Ablation")
        if 'advanced_ablation' in self.results_summary:
            best = self.results_summary['advanced_ablation']['best_config']
            report_lines.append(f"- Best Combined Win Rate: {best.get('combined_win_rate', 0):.1%}")
            report_lines.append(f"- Number of Features: {best.get('num_features', 0)}")
        
        report_lines.append("\n### Trading Evaluation")
        if 'trading_evaluation' in self.results_summary:
            metrics = self.results_summary['trading_evaluation']['main_metrics']
            report_lines.append(f"- Combined Win Rate: {metrics.get('combined_win_rate', 0):.1%}")
            report_lines.append(f"- Total Return: {metrics.get('total_return', 0):.4f}")
            report_lines.append(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            report_lines.append(f"- Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
        
        # Best Configuration
        report_lines.append("\n## BEST CONFIGURATION")
        if self.best_configuration:
            report_lines.append(f"\n- Buy Threshold: {self.best_configuration['buy_threshold_pct']}%")
            report_lines.append(f"- Sell Threshold: {self.best_configuration['sell_threshold_pct']}%")
            report_lines.append(f"- Number of Features: {len(self.best_configuration['features'])}")
            report_lines.append(f"\nFeatures:")
            for i, feature in enumerate(self.best_configuration['features'][:20], 1):
                report_lines.append(f"  {i}. {feature}")
            if len(self.best_configuration['features']) > 20:
                report_lines.append(f"  ... va {len(self.best_configuration['features']) - 20} features khac")
        
        # Recommendations
        report_lines.append("\n## KHUYEN NGHI")
        for rec in self.recommendations:
            report_lines.append(f"\n- {rec}")
        
        # Luu report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Da luu bao cao vao {report_path}")
    
    def save_best_configuration(self):
        """Luu best configuration vao JSON"""
        config_path = config.OUTPUT_FILES['best_configuration']
        
        config_dict = {
            'buy_threshold_pct': self.best_configuration.get('buy_threshold_pct', 75),
            'sell_threshold_pct': self.best_configuration.get('sell_threshold_pct', 75),
            'features': self.best_configuration.get('features', []),
            'metrics': self.best_configuration.get('metrics', {})
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Da luu best configuration vao {config_path}")
    
    def save_recommendations(self):
        """Luu khuyen nghi vao file text"""
        rec_path = config.OUTPUT_FILES['recommendations']
        
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write("KHUYEN NGHI IMPLEMENTATION\n")
            f.write("="*60 + "\n\n")
            for rec in self.recommendations:
                f.write(f"{rec}\n")
        
        print(f"Da luu khuyen nghi vao {rec_path}")
    
    def run_all_tests(self):
        """Chay tat ca cac test"""
        print("="*60)
        print("COMPREHENSIVE TEST RUNNER")
        print("="*60)
        print("\nBat dau chay tat ca cac test...")
        
        try:
            # Phase 1: Threshold Optimization
            best_combined, best_weighted, best_sharpe = self.run_threshold_optimization()
            
            # Su dung best threshold cho cac test tiep theo
            buy_threshold_pct = int(best_combined['buy_percentile'])
            sell_threshold_pct = int(best_combined['sell_percentile'])
            
            # Phase 2: Feature Combination
            best_feature_combo = self.run_feature_combination(buy_threshold_pct, sell_threshold_pct)
            
            # Phase 3: Advanced Ablation
            best_ablation = self.run_advanced_ablation(buy_threshold_pct, sell_threshold_pct)
            
            # Phase 4: Trading Evaluation
            # Xac dinh best features
            best_features = None
            if best_feature_combo is not None and 'test_features' in best_feature_combo:
                best_features = best_feature_combo['test_features'].split(',')
            elif best_ablation is not None and 'test_features' in best_ablation:
                best_features = best_ablation['test_features'].split(',')
            else:
                # Fallback: su dung no_trend
                from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete
                predictor = NVDA_MultiStock_Complete()
                df, feature_cols = predictor.load_multi_stock_data(config.DATA_DIR)
                no_trend = [f for f in feature_cols if not any(x in f.lower() for x in ['sma', 'price_vs'])]
                best_features = no_trend
            
            if best_features:
                feature_to_idx = {f: i for i, f in enumerate(best_features)}
                feature_indices = list(range(len(best_features)))
                
                # Load data de lay feature indices chinh xac
                predictor = NVDA_MultiStock_Complete()
                df, feature_cols = predictor.load_multi_stock_data(config.DATA_DIR)
                feature_to_idx_full = {f: i for i, f in enumerate(feature_cols)}
                feature_indices = [feature_to_idx_full[f] for f in best_features if f in feature_to_idx_full]
                
                self.run_trading_evaluation(feature_indices, best_features, buy_threshold_pct, sell_threshold_pct)
            
            # Phase 5: Determine best configuration
            self.determine_best_configuration()
            
            # Generate recommendations
            self.generate_recommendations()
            
            # Save everything
            self.generate_report()
            self.save_best_configuration()
            self.save_recommendations()
            
            print("\n" + "="*60)
            print("HOAN THANH TAT CA CAC TEST")
            print("="*60)
            print("\nKet qua da duoc luu vao:")
            print(f"  - {config.OUTPUT_FILES['comprehensive_report']}")
            print(f"  - {config.OUTPUT_FILES['best_configuration']}")
            print(f"  - {config.OUTPUT_FILES['recommendations']}")
            
        except Exception as e:
            print(f"\nLoi khi chay tests: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main workflow"""
    runner = ComprehensiveTestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()

