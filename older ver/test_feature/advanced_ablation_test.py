#!/usr/bin/env python3
"""
Advanced Ablation Testing
Ablation test chi tiet hon voi cac nhom feature nho va combinations
Bao gom cross-validation de dam bao tinh on dinh
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v4_multistock'))
sys.path.append('..')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete, LSTMRegressor
import config
import torch

# Hien thi thong tin device
if config.DEVICE:
    print(f"Su dung device: {config.DEVICE}")
    if config.USE_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Canh bao: Dang su dung CPU. De su dung GPU, chay: python install_pytorch_cuda.py")

class AdvancedAblationTester:
    def __init__(self, data_dir=None):
        """Khoi tao voi data directory"""
        self.data_dir = data_dir or config.DATA_DIR
        self.predictor = NVDA_MultiStock_Complete(
            sequence_length=config.SEQUENCE_LENGTH,
            horizon=config.HORIZON
        )
        self.results = []
        self.feature_contributions = {}
        
    def load_data(self):
        """Load va chuan bi data"""
        print("Dang load du lieu multi-stock...")
        df, feature_cols = self.predictor.load_multi_stock_data(self.data_dir)
        
        (X_train, y_train_reg, y_train_cls,
         X_test, y_test_reg, y_test_cls) = self.predictor.split_data_transfer_learning(df, feature_cols)
        
        print(f"  Train: {X_train.shape[0]} sequences")
        print(f"  Test: {X_test.shape[0]} sequences (NVDA only)")
        
        return X_train, X_test, y_train_reg, y_test_reg, feature_cols
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_indices,
                          buy_threshold_pct=75, sell_threshold_pct=75, epochs=None):
        """Train model voi feature cu the va danh gia"""
        epochs = epochs or config.EPOCHS
        
        # Chon features
        X_train_sub = X_train[:, :, feature_indices]
        X_test_sub = X_test[:, :, feature_indices]
        
        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_flat = X_train_sub.reshape(-1, X_train_sub.shape[-1])
        X_test_flat = X_test_sub.reshape(-1, X_test_sub.shape[-1])
        
        X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train_sub.shape)
        X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test_sub.shape)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        
        # Tao model va chuyen sang GPU neu co
        model = LSTMRegressor(input_size=len(feature_indices))
        device = config.DEVICE if config.DEVICE else torch.device("cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Chuyen data sang GPU
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        # Train voi early stopping
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    break
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
            if y_pred_scaled.ndim > 1:
                y_pred_scaled = y_pred_scaled.flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Tinh metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Trading metrics voi threshold
        buy_threshold = np.percentile(y_pred[y_pred > 0], buy_threshold_pct) if np.any(y_pred > 0) else np.percentile(np.abs(y_pred), buy_threshold_pct)
        sell_threshold = -np.percentile(-y_pred[y_pred < 0], 100 - sell_threshold_pct) if np.any(y_pred < 0) else -np.percentile(np.abs(y_pred), 100 - sell_threshold_pct)
        
        if np.isnan(buy_threshold) or buy_threshold <= 0:
            buy_threshold = np.percentile(np.abs(y_pred), buy_threshold_pct)
        if np.isnan(sell_threshold) or sell_threshold >= 0:
            sell_threshold = -np.percentile(np.abs(y_pred), 100 - sell_threshold_pct)
        
        signals = np.zeros(len(y_pred))
        signals[y_pred > buy_threshold] = 2
        signals[y_pred < sell_threshold] = 0
        
        buy_returns = y_test[signals == 2]
        sell_returns = y_test[signals == 0]
        
        buy_win_rate = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
        sell_win_rate = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
        combined_win_rate = (buy_win_rate + sell_win_rate) / 2 if (len(buy_returns) > 0 or len(sell_returns) > 0) else 0
        coverage = (len(buy_returns) + len(sell_returns)) / len(signals)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'combined_win_rate': combined_win_rate,
            'coverage': coverage,
            'num_features': len(feature_indices),
            'num_buy_signals': len(buy_returns),
            'num_sell_signals': len(sell_returns)
        }
    
    def cross_validate(self, X_train, y_train, feature_indices, 
                       buy_threshold_pct=75, sell_threshold_pct=75, n_splits=None):
        """Cross-validation de danh gia tinh on dinh"""
        n_splits = n_splits or config.CV_FOLDS
        
        # Chuan bi data cho CV
        X_train_sub = X_train[:, :, feature_indices]
        X_train_flat = X_train_sub.reshape(-1, X_train_sub.shape[-1])
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.CV_RANDOM_STATE)
        
        cv_results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_flat)):
            # Chia data
            X_train_fold = X_train_sub[train_idx]
            X_val_fold = X_train_sub[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]
            
            # Train va evaluate
            metrics = self.train_and_evaluate(
                X_train_fold, X_val_fold, y_train_fold, y_val_fold, feature_indices,
                buy_threshold_pct, sell_threshold_pct, epochs=50  # It hon cho CV
            )
            metrics['fold'] = fold
            cv_results.append(metrics)
        
        # Tinh trung binh va std
        cv_df = pd.DataFrame(cv_results)
        summary = {
            'mean_buy_win_rate': cv_df['buy_win_rate'].mean(),
            'std_buy_win_rate': cv_df['buy_win_rate'].std(),
            'mean_sell_win_rate': cv_df['sell_win_rate'].mean(),
            'std_sell_win_rate': cv_df['sell_win_rate'].std(),
            'mean_combined_win_rate': cv_df['combined_win_rate'].mean(),
            'std_combined_win_rate': cv_df['combined_win_rate'].std(),
            'mean_rmse': cv_df['rmse'].mean(),
            'std_rmse': cv_df['rmse'].std(),
            'stability_score': 1.0 - (cv_df['combined_win_rate'].std() / (cv_df['combined_win_rate'].mean() + 1e-6))
        }
        
        return summary
    
    def single_feature_ablation(self, X_train, X_test, y_train, y_test, feature_cols,
                               base_features, buy_threshold_pct=75, sell_threshold_pct=75):
        """Test loai bo tung feature don le"""
        print("\nDang chay Single Feature Ablation...")
        
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        base_indices = [feature_to_idx[f] for f in base_features if f in feature_to_idx]
        
        # Baseline performance
        baseline_metrics = self.train_and_evaluate(
            X_train, X_test, y_train, y_test, base_indices,
            buy_threshold_pct, sell_threshold_pct
        )
        baseline_wr = baseline_metrics['combined_win_rate']
        
        results = []
        for feature in base_features:
            if feature not in feature_to_idx:
                continue
            
            test_features = [f for f in base_features if f != feature]
            test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
            
            if len(test_indices) == 0:
                continue
            
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices,
                buy_threshold_pct, sell_threshold_pct
            )
            
            impact = baseline_wr - metrics['combined_win_rate']
            metrics['feature_removed'] = feature
            metrics['impact'] = impact
            metrics['test_type'] = 'single_feature_ablation'
            metrics['base_features'] = ','.join(base_features)
            results.append(metrics)
            
            # Luu feature contribution
            self.feature_contributions[feature] = {
                'impact': impact,
                'baseline_wr': baseline_wr,
                'without_wr': metrics['combined_win_rate']
            }
            
            if config.VERBOSE:
                print(f"  Loai bo {feature}: Impact = {impact:+.3f}, WR = {metrics['combined_win_rate']:.1%}")
        
        return results
    
    def test_small_feature_groups(self, X_train, X_test, y_train, y_test, feature_cols,
                                  buy_threshold_pct=75, sell_threshold_pct=75):
        """Test cac nhom feature nho"""
        print("\nDang test cac nhom feature nho...")
        
        groups = {}
        for group_name, group_features in config.FEATURE_GROUPS.items():
            existing_features = [f for f in group_features if f in feature_cols]
            if existing_features:
                groups[group_name] = existing_features
        
        results = []
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        
        # Test tung nhom
        for group_name, group_features in groups.items():
            if len(group_features) < 2:
                continue
            
            # Test chi voi nhom nay
            test_indices = [feature_to_idx[f] for f in group_features if f in feature_to_idx]
            
            if len(test_indices) == 0:
                continue
            
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices,
                buy_threshold_pct, sell_threshold_pct
            )
            
            metrics['group_name'] = group_name
            metrics['group_features'] = ','.join(group_features)
            metrics['test_type'] = 'small_group_test'
            results.append(metrics)
            
            if config.VERBOSE:
                print(f"  {group_name}: WR = {metrics['combined_win_rate']:.1%}")
        
        return results
    
    def test_momentum_subgroups(self, X_train, X_test, y_train, y_test, feature_cols,
                               buy_threshold_pct=75, sell_threshold_pct=75):
        """Test cac nhom con cua momentum (rsi vs macd)"""
        print("\nDang test momentum subgroups...")
        
        groups = {}
        for group_name, group_features in config.FEATURE_GROUPS.items():
            existing_features = [f for f in group_features if f in feature_cols]
            if existing_features:
                groups[group_name] = existing_features
        
        momentum_features = groups.get('momentum', [])
        if len(momentum_features) < 2:
            return []
        
        # Chia thanh rsi va macd
        rsi_features = [f for f in momentum_features if 'rsi' in f.lower()]
        macd_features = [f for f in momentum_features if 'macd' in f.lower()]
        
        results = []
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        
        # Test chi voi RSI
        if rsi_features:
            test_indices = [feature_to_idx[f] for f in rsi_features if f in feature_to_idx]
            if test_indices:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices,
                    buy_threshold_pct, sell_threshold_pct
                )
                metrics['subgroup'] = 'rsi_only'
                metrics['test_type'] = 'momentum_subgroup'
                results.append(metrics)
        
        # Test chi voi MACD
        if macd_features:
            test_indices = [feature_to_idx[f] for f in macd_features if f in feature_to_idx]
            if test_indices:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices,
                    buy_threshold_pct, sell_threshold_pct
                )
                metrics['subgroup'] = 'macd_only'
                metrics['test_type'] = 'momentum_subgroup'
                results.append(metrics)
        
        return results
    
    def test_volatility_subgroups(self, X_train, X_test, y_train, y_test, feature_cols,
                                 buy_threshold_pct=75, sell_threshold_pct=75):
        """Test cac nhom con cua volatility (atr vs bb features)"""
        print("\nDang test volatility subgroups...")
        
        groups = {}
        for group_name, group_features in config.FEATURE_GROUPS.items():
            existing_features = [f for f in group_features if f in feature_cols]
            if existing_features:
                groups[group_name] = existing_features
        
        volatility_features = groups.get('volatility', [])
        if len(volatility_features) < 2:
            return []
        
        # Chia thanh atr va bb
        atr_features = [f for f in volatility_features if 'atr' in f.lower()]
        bb_features = [f for f in volatility_features if 'bb' in f.lower()]
        
        results = []
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        
        # Test chi voi ATR
        if atr_features:
            test_indices = [feature_to_idx[f] for f in atr_features if f in feature_to_idx]
            if test_indices:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices,
                    buy_threshold_pct, sell_threshold_pct
                )
                metrics['subgroup'] = 'atr_only'
                metrics['test_type'] = 'volatility_subgroup'
                results.append(metrics)
        
        # Test chi voi BB
        if bb_features:
            test_indices = [feature_to_idx[f] for f in bb_features if f in feature_to_idx]
            if test_indices:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices,
                    buy_threshold_pct, sell_threshold_pct
                )
                metrics['subgroup'] = 'bb_only'
                metrics['test_type'] = 'volatility_subgroup'
                results.append(metrics)
        
        return results
    
    def test_return_timeframes(self, X_train, X_test, y_train, y_test, feature_cols,
                              buy_threshold_pct=75, sell_threshold_pct=75):
        """Test cac return features voi timeframe khac nhau"""
        print("\nDang test return timeframes...")
        
        groups = {}
        for group_name, group_features in config.FEATURE_GROUPS.items():
            existing_features = [f for f in group_features if f in feature_cols]
            if existing_features:
                groups[group_name] = existing_features
        
        return_features = groups.get('returns', [])
        if len(return_features) < 2:
            return []
        
        # Chia thanh short-term va long-term
        short_term = [f for f in return_features if any(x in f.lower() for x in ['daily', 'change', '3d', '5d'])]
        long_term = [f for f in return_features if any(x in f.lower() for x in ['10d', '20d'])]
        
        results = []
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        
        # Test chi voi short-term
        if short_term:
            test_indices = [feature_to_idx[f] for f in short_term if f in feature_to_idx]
            if test_indices:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices,
                    buy_threshold_pct, sell_threshold_pct
                )
                metrics['timeframe'] = 'short_term'
                metrics['test_type'] = 'return_timeframe'
                results.append(metrics)
        
        # Test chi voi long-term
        if long_term:
            test_indices = [feature_to_idx[f] for f in long_term if f in feature_to_idx]
            if test_indices:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices,
                    buy_threshold_pct, sell_threshold_pct
                )
                metrics['timeframe'] = 'long_term'
                metrics['test_type'] = 'return_timeframe'
                results.append(metrics)
        
        return results
    
    def test_feature_combinations(self, X_train, X_test, y_train, y_test, feature_cols,
                                 buy_threshold_pct=75, sell_threshold_pct=75):
        """Test cac combinations: no_trend + no_volume, no_trend + no_structure, etc."""
        print("\nDang test feature combinations...")
        
        # Lay no_trend features
        no_trend = [f for f in feature_cols if not any(x in f.lower() for x in ['sma', 'price_vs'])]
        
        # Cac nhom de loai bo
        groups_to_remove = ['volume', 'structure', 'position']
        
        results = []
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        
        for group_name in groups_to_remove:
            group_features = config.FEATURE_GROUPS.get(group_name, [])
            if not group_features:
                continue
            
            # Loai bo nhom nay khoi no_trend
            test_features = [f for f in no_trend if f not in group_features]
            test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
            
            if len(test_indices) == 0:
                continue
            
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices,
                buy_threshold_pct, sell_threshold_pct
            )
            
            metrics['combination'] = f'no_trend_no_{group_name}'
            metrics['test_type'] = 'feature_combination'
            results.append(metrics)
            
            if config.VERBOSE:
                print(f"  no_trend + no_{group_name}: WR = {metrics['combined_win_rate']:.1%}")
        
        return results
    
    def run_comprehensive_test(self, buy_threshold_pct=75, sell_threshold_pct=75):
        """Chay tat ca cac test"""
        print("="*60)
        print("ADVANCED ABLATION TESTING")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test, feature_cols = self.load_data()
        
        # Lay no_trend features (baseline)
        no_trend_features = [f for f in feature_cols if not any(x in f.lower() for x in ['sma', 'price_vs'])]
        print(f"\nBaseline: no_trend features ({len(no_trend_features)} features)")
        
        all_results = []
        
        # Single feature ablation
        single_results = self.single_feature_ablation(
            X_train, X_test, y_train, y_test, feature_cols, no_trend_features,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(single_results)
        
        # Test small feature groups
        small_group_results = self.test_small_feature_groups(
            X_train, X_test, y_train, y_test, feature_cols,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(small_group_results)
        
        # Test momentum subgroups
        momentum_results = self.test_momentum_subgroups(
            X_train, X_test, y_train, y_test, feature_cols,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(momentum_results)
        
        # Test volatility subgroups
        volatility_results = self.test_volatility_subgroups(
            X_train, X_test, y_train, y_test, feature_cols,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(volatility_results)
        
        # Test return timeframes
        return_results = self.test_return_timeframes(
            X_train, X_test, y_train, y_test, feature_cols,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(return_results)
        
        # Test feature combinations
        combination_results = self.test_feature_combinations(
            X_train, X_test, y_train, y_test, feature_cols,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(combination_results)
        
        # Cross-validation cho best configurations
        if len(all_results) > 0:
            print("\nDang chay Cross-Validation cho cac configuration tot nhat...")
            results_df = pd.DataFrame(all_results)
            top_configs = results_df.nlargest(3, 'combined_win_rate')
            
            feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
            for idx, row in top_configs.iterrows():
                if 'test_features' in row and pd.notna(row['test_features']):
                    test_features = row['test_features'].split(',')
                    test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
                    
                    cv_summary = self.cross_validate(
                        X_train, y_train, test_indices,
                        buy_threshold_pct, sell_threshold_pct
                    )
                    
                    cv_summary['config_index'] = idx
                    cv_summary['test_features'] = row['test_features']
                    all_results.append(cv_summary)
        
        self.results = pd.DataFrame(all_results)
        
        # Tim best configuration
        if len(self.results) > 0:
            best = self.results.loc[self.results['combined_win_rate'].idxmax()]
            print("\n" + "="*60)
            print("KET QUA TOI UU")
            print("="*60)
            print(f"Best Combined Win Rate: {best['combined_win_rate']:.1%}")
            print(f"Buy WR: {best['buy_win_rate']:.1%}, Sell WR: {best['sell_win_rate']:.1%}")
            print(f"RMSE: {best['rmse']:.4f}")
            print(f"Number of features: {best['num_features']}")
        
        # Tao feature contribution analysis
        self.create_feature_contribution_analysis()
        
        return self.results
    
    def create_feature_contribution_analysis(self):
        """Tao phan tich dong gop cua tung feature"""
        if not self.feature_contributions:
            return
        
        contribution_df = pd.DataFrame([
            {
                'feature': k,
                'impact': v['impact'],
                'baseline_wr': v['baseline_wr'],
                'without_wr': v['without_wr']
            }
            for k, v in self.feature_contributions.items()
        ])
        contribution_df = contribution_df.sort_values('impact', ascending=False)
        
        save_path = config.OUTPUT_FILES['feature_contribution']
        contribution_df.to_csv(save_path, index=False)
        print(f"\nDa luu feature contribution analysis vao {save_path}")
    
    def save_results(self, save_path=None):
        """Luu ket qua"""
        if len(self.results) == 0:
            print("Chua co ket qua de luu")
            return
        
        save_path = save_path or config.OUTPUT_FILES['advanced_ablation']
        self.results.to_csv(save_path, index=False)
        print(f"Da luu ket qua vao {save_path}")


def main():
    """Main workflow"""
    tester = AdvancedAblationTester()
    
    # Chay comprehensive test
    results = tester.run_comprehensive_test()
    
    # Luu ket qua
    tester.save_results()
    
    print("\nHoan thanh advanced ablation testing!")


if __name__ == "__main__":
    main()

