#!/usr/bin/env python3
"""
Quick Correlation-based Reduction
Loai bo features redundant dua tren correlation groups
Training toi uu: epochs=50, patience=10
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete, LSTMRegressor
import config

# Load feature set tu v5
v5_path = os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v5_multistock', 'nvda_lstm_v5_multistock.py')
sys.path.insert(0, os.path.dirname(v5_path))
try:
    from nvda_lstm_v5_multistock import load_optimized_features
except:
    OPT_CONF = os.path.join(os.path.dirname(__file__), 'optimized_feature_config.csv')
    def load_optimized_features():
        if os.path.exists(OPT_CONF):
            df = pd.read_csv(OPT_CONF)
            if not df.empty and 'features' in df.columns:
                raw = df.iloc[0]['features']
                try:
                    return eval(raw)
                except:
                    pass
        return [
            'rsi14','macd','macd_bullish','macd_signal','macd_hist',
            'atr','bb_bandwidth','bb_percent',
            'volume_ratio','obv','volume_sma20',
            'daily_return','price_change','return_3d','return_5d','return_10d','return_20d',
            'hl_spread_pct','oc_spread','oc_spread_pct',
            'bb_squeeze','rsi_overbought','rsi_oversold',
            'sox_beta','sox_correlation'
        ]

class QuickCorrelationReduction:
    def __init__(self, data_dir=None):
        """Khoi tao voi data directory"""
        self.data_dir = data_dir or config.DATA_DIR
        self.predictor = NVDA_MultiStock_Complete(
            sequence_length=config.SEQUENCE_LENGTH,
            horizon=config.HORIZON
        )
        self.v5_features = load_optimized_features()
        self.results = []
        
        # Dinh nghia cac nhom features co the redundant
        self.feature_groups = {
            'macd': ['macd', 'macd_signal', 'macd_hist', 'macd_bullish'],
            'return': ['daily_return', 'price_change', 'return_3d', 'return_5d', 'return_10d', 'return_20d'],
            'spread': ['hl_spread_pct', 'oc_spread', 'oc_spread_pct'],
            'bb': ['bb_bandwidth', 'bb_percent', 'bb_squeeze'],
            'rsi': ['rsi14', 'rsi_overbought', 'rsi_oversold'],
            'volume': ['volume_ratio', 'obv', 'volume_sma20'],
            'sector': ['sox_beta', 'sox_correlation']
        }
        
    def load_data(self):
        """Load va chuan bi data"""
        print("Dang load data...")
        df, all_features = self.predictor.load_multi_stock_data(self.data_dir)
        
        # Filter theo v5 features
        available = [f for f in self.v5_features if f in all_features]
        missing = set(self.v5_features) - set(available)
        if missing:
            print(f"Canh bao: Thieu features: {missing}")
        
        print(f"Su dung {len(available)} features tu v5")
        
        (X_train, y_train_reg, y_train_cls,
         X_test, y_test_reg, y_test_cls) = self.predictor.split_data_transfer_learning(df, available)
        
        print(f"  Train: {X_train.shape[0]} sequences")
        print(f"  Test: {X_test.shape[0]} sequences (NVDA only)")
        
        return X_train, X_test, y_train_reg, y_test_reg, available
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_indices,
                          buy_threshold_pct=75, sell_threshold_pct=75, epochs=50):
        """Train model voi feature cu the va danh gia (toi uu cho speed)"""
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Chuyen data sang GPU
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        # Train voi early stopping (toi uu: epochs=50, patience=10)
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
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
                if patience_counter >= patience:
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
    
    def reduce_group_features(self, feature_cols, group_name, keep_count=1):
        """Giam so luong features trong mot nhom, chi giu keep_count features tot nhat"""
        group_features = self.feature_groups.get(group_name, [])
        existing = [f for f in group_features if f in feature_cols]
        
        if len(existing) <= keep_count:
            return existing
        
        # Neu co VIF data, uu tien giu features co VIF thap
        vif_path = os.path.join(config.RESULTS_DIR, 'quick_vif_analysis.csv')
        if os.path.exists(vif_path):
            vif_df = pd.read_csv(vif_path)
            vif_dict = dict(zip(vif_df['feature'], vif_df['vif']))
            existing.sort(key=lambda x: vif_dict.get(x, 999))
            return existing[:keep_count]
        else:
            # Neu khong co VIF, giu features dau tien
            return existing[:keep_count]
    
    def run_reduction_tests(self):
        """Chay cac test reduction theo correlation groups"""
        print("="*60)
        print("QUICK CORRELATION-BASED REDUCTION")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test, feature_cols = self.load_data()
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        
        # Test baseline
        print("\nDang test baseline (tat ca features)...")
        baseline_indices = list(range(len(feature_cols)))
        baseline_metrics = self.train_and_evaluate(
            X_train, X_test, y_train, y_test, baseline_indices, epochs=50
        )
        baseline_metrics['test_type'] = 'baseline'
        baseline_metrics['reduction_strategy'] = 'none'
        baseline_metrics['kept_features'] = ','.join(feature_cols)
        self.results.append(baseline_metrics)
        print(f"  Baseline: WR={baseline_metrics['combined_win_rate']:.1%}, Features={baseline_metrics['num_features']}")
        
        # Test 1: Giam MACD group (chi giu macd va macd_bullish)
        print("\nDang test giam MACD group...")
        test_features = feature_cols.copy()
        macd_group = self.feature_groups['macd']
        macd_existing = [f for f in macd_group if f in test_features]
        if len(macd_existing) > 2:
            # Loai bo macd_signal va macd_hist, giu macd va macd_bullish
            to_remove = [f for f in macd_existing if f not in ['macd', 'macd_bullish']]
            test_features = [f for f in test_features if f not in to_remove]
            test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
            
            if len(test_indices) > 0:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices, epochs=50
                )
                metrics['test_type'] = 'reduce_macd'
                metrics['reduction_strategy'] = f'remove_{",".join(to_remove)}'
                metrics['kept_features'] = ','.join(test_features)
                self.results.append(metrics)
                print(f"  Reduce MACD: WR={metrics['combined_win_rate']:.1%}, Features={metrics['num_features']}")
        
        # Test 2: Giam Return group (chi giu daily_return, price_change, return_5d)
        print("\nDang test giam Return group...")
        test_features = feature_cols.copy()
        return_group = self.feature_groups['return']
        return_existing = [f for f in return_group if f in test_features]
        if len(return_existing) > 3:
            # Chi giu daily_return, price_change, return_5d
            to_keep = ['daily_return', 'price_change', 'return_5d']
            to_remove = [f for f in return_existing if f not in to_keep]
            test_features = [f for f in test_features if f not in to_remove]
            test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
            
            if len(test_indices) > 0:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices, epochs=50
                )
                metrics['test_type'] = 'reduce_returns'
                metrics['reduction_strategy'] = f'remove_{",".join(to_remove)}'
                metrics['kept_features'] = ','.join(test_features)
                self.results.append(metrics)
                print(f"  Reduce Returns: WR={metrics['combined_win_rate']:.1%}, Features={metrics['num_features']}")
        
        # Test 3: Giam Spread group (chi giu hl_spread_pct)
        print("\nDang test giam Spread group...")
        test_features = feature_cols.copy()
        spread_group = self.feature_groups['spread']
        spread_existing = [f for f in spread_group if f in test_features]
        if len(spread_existing) > 1:
            # Chi giu hl_spread_pct
            to_remove = [f for f in spread_existing if f != 'hl_spread_pct']
            test_features = [f for f in test_features if f not in to_remove]
            test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
            
            if len(test_indices) > 0:
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices, epochs=50
                )
                metrics['test_type'] = 'reduce_spread'
                metrics['reduction_strategy'] = f'remove_{",".join(to_remove)}'
                metrics['kept_features'] = ','.join(test_features)
                self.results.append(metrics)
                print(f"  Reduce Spread: WR={metrics['combined_win_rate']:.1%}, Features={metrics['num_features']}")
        
        # Test 4: Combined reduction (MACD + Returns + Spread)
        print("\nDang test combined reduction (MACD + Returns + Spread)...")
        test_features = feature_cols.copy()
        
        # Giam MACD
        macd_existing = [f for f in self.feature_groups['macd'] if f in test_features]
        if len(macd_existing) > 2:
            to_remove_macd = [f for f in macd_existing if f not in ['macd', 'macd_bullish']]
            test_features = [f for f in test_features if f not in to_remove_macd]
        
        # Giam Returns
        return_existing = [f for f in self.feature_groups['return'] if f in test_features]
        if len(return_existing) > 3:
            to_keep_returns = ['daily_return', 'price_change', 'return_5d']
            to_remove_returns = [f for f in return_existing if f not in to_keep_returns]
            test_features = [f for f in test_features if f not in to_remove_returns]
        
        # Giam Spread
        spread_existing = [f for f in self.feature_groups['spread'] if f in test_features]
        if len(spread_existing) > 1:
            to_remove_spread = [f for f in spread_existing if f != 'hl_spread_pct']
            test_features = [f for f in test_features if f not in to_remove_spread]
        
        test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
        
        if len(test_indices) > 0 and len(test_indices) < len(feature_cols):
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices, epochs=50
            )
            metrics['test_type'] = 'combined_reduction'
            metrics['reduction_strategy'] = 'macd+returns+spread'
            metrics['kept_features'] = ','.join(test_features)
            self.results.append(metrics)
            print(f"  Combined: WR={metrics['combined_win_rate']:.1%}, Features={metrics['num_features']}")
        
        # Tim best configuration
        results_df = pd.DataFrame(self.results)
        if len(results_df) > 0:
            best = results_df.loc[results_df['combined_win_rate'].idxmax()]
            
            print("\n" + "="*60)
            print("KET QUA TOI UU")
            print("="*60)
            print(f"Best Configuration: {best['test_type']}")
            print(f"  Combined Win Rate: {best['combined_win_rate']:.1%}")
            print(f"  Buy WR: {best['buy_win_rate']:.1%}, Sell WR: {best['sell_win_rate']:.1%}")
            print(f"  RMSE: {best['rmse']:.4f}")
            print(f"  Number of features: {best['num_features']}")
            if best['reduction_strategy'] != 'none':
                print(f"  Reduction: {best['reduction_strategy']}")
        
        return results_df
    
    def save_results(self, results_df):
        """Luu ket qua"""
        results_dir = os.path.join(os.path.dirname(__file__), config.RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        
        save_path = os.path.join(results_dir, 'quick_correlation_reduction_results.csv')
        results_df.to_csv(save_path, index=False)
        print(f"\nDa luu ket qua vao {save_path}")


def main():
    """Main workflow"""
    tester = QuickCorrelationReduction()
    results = tester.run_reduction_tests()
    tester.save_results(results)
    
    print("\nHoan thanh quick correlation reduction test!")


if __name__ == "__main__":
    main()

