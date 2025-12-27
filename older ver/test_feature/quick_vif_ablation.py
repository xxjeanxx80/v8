#!/usr/bin/env python3
"""
Quick VIF-based Ablation Test
Test nhanh bang cach loai bo cac features co VIF cao nhat
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

class QuickVIFAblation:
    def __init__(self, data_dir=None):
        """Khoi tao voi data directory"""
        self.data_dir = data_dir or config.DATA_DIR
        self.predictor = NVDA_MultiStock_Complete(
            sequence_length=config.SEQUENCE_LENGTH,
            horizon=config.HORIZON
        )
        self.v5_features = load_optimized_features()
        self.vif_data = None
        self.results = []
        
    def load_vif_data(self):
        """Load VIF data tu quick_multicollinearity_test"""
        vif_path = os.path.join(config.RESULTS_DIR, 'quick_vif_analysis.csv')
        if os.path.exists(vif_path):
            self.vif_data = pd.read_csv(vif_path)
            print(f"Da load VIF data tu {vif_path}")
            return True
        else:
            print("Chua co VIF data, se tinh toan sau khi load data")
            return False
    
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
    
    def calculate_vif_for_features(self, df, feature_cols):
        """Tinh VIF cho features neu chua co"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        
        X = df[feature_cols].dropna()
        X = X.loc[:, X.var() > 1e-6]
        feature_cols = list(X.columns)
        
        X_const = add_constant(X)
        vif_data = []
        
        for i, feature in enumerate(X.columns):
            try:
                vif = variance_inflation_factor(X_const.values, i+1)
                vif_data.append({
                    'feature': feature,
                    'vif': vif if not np.isinf(vif) and not np.isnan(vif) else 999.0
                })
            except:
                vif_data.append({'feature': feature, 'vif': 999.0})
        
        return pd.DataFrame(vif_data).sort_values('vif', ascending=False)
    
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
        patience = 10  # Giam tu 20 xuong 10
        
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
    
    def run_ablation(self):
        """Chay ablation test loai bo features co VIF cao"""
        print("="*60)
        print("QUICK VIF-BASED ABLATION TEST")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test, feature_cols = self.load_data()
        
        # Load hoac tinh VIF
        if not self.load_vif_data():
            # Tinh VIF neu chua co
            print("\nDang tinh VIF...")
            df, _ = self.predictor.load_multi_stock_data(self.data_dir)
            self.vif_data = self.calculate_vif_for_features(df, feature_cols)
        
        # Sap xep features theo VIF (tu cao xuong thap)
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        vif_sorted = self.vif_data.sort_values('vif', ascending=False)
        
        print(f"\nFeatures sap xep theo VIF (tu cao xuong thap):")
        for i, row in vif_sorted.head(10).iterrows():
            print(f"  {row['feature']}: VIF = {row['vif']:.2f}")
        
        # Test baseline (tat ca features)
        print("\nDang test baseline (tat ca features)...")
        baseline_indices = list(range(len(feature_cols)))
        baseline_metrics = self.train_and_evaluate(
            X_train, X_test, y_train, y_test, baseline_indices, epochs=50
        )
        baseline_metrics['test_type'] = 'baseline'
        baseline_metrics['removed_features'] = ''
        baseline_metrics['avg_vif'] = self.vif_data['vif'].mean()
        self.results.append(baseline_metrics)
        print(f"  Baseline: WR={baseline_metrics['combined_win_rate']:.1%}, RMSE={baseline_metrics['rmse']:.4f}, Avg VIF={baseline_metrics['avg_vif']:.2f}")
        
        # Test loai bo top 3 VIF features
        print("\nDang test loai bo top 3 VIF features...")
        top3_vif = vif_sorted.head(3)['feature'].tolist()
        test_features = [f for f in feature_cols if f not in top3_vif]
        test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
        
        if len(test_indices) > 0:
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices, epochs=50
            )
            metrics['test_type'] = 'remove_top3_vif'
            metrics['removed_features'] = ','.join(top3_vif)
            remaining_vif = self.vif_data[self.vif_data['feature'].isin(test_features)]
            metrics['avg_vif'] = remaining_vif['vif'].mean() if len(remaining_vif) > 0 else 0
            self.results.append(metrics)
            print(f"  Remove top 3: WR={metrics['combined_win_rate']:.1%}, RMSE={metrics['rmse']:.4f}, Avg VIF={metrics['avg_vif']:.2f}")
        
        # Test loai bo top 5 VIF features
        print("\nDang test loai bo top 5 VIF features...")
        top5_vif = vif_sorted.head(5)['feature'].tolist()
        test_features = [f for f in feature_cols if f not in top5_vif]
        test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
        
        if len(test_indices) > 0:
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices, epochs=50
            )
            metrics['test_type'] = 'remove_top5_vif'
            metrics['removed_features'] = ','.join(top5_vif)
            remaining_vif = self.vif_data[self.vif_data['feature'].isin(test_features)]
            metrics['avg_vif'] = remaining_vif['vif'].mean() if len(remaining_vif) > 0 else 0
            self.results.append(metrics)
            print(f"  Remove top 5: WR={metrics['combined_win_rate']:.1%}, RMSE={metrics['rmse']:.4f}, Avg VIF={metrics['avg_vif']:.2f}")
        
        # Test loai bo top 7 VIF features
        print("\nDang test loai bo top 7 VIF features...")
        top7_vif = vif_sorted.head(7)['feature'].tolist()
        test_features = [f for f in feature_cols if f not in top7_vif]
        test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
        
        if len(test_indices) > 0:
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices, epochs=50
            )
            metrics['test_type'] = 'remove_top7_vif'
            metrics['removed_features'] = ','.join(top7_vif)
            remaining_vif = self.vif_data[self.vif_data['feature'].isin(test_features)]
            metrics['avg_vif'] = remaining_vif['vif'].mean() if len(remaining_vif) > 0 else 0
            self.results.append(metrics)
            print(f"  Remove top 7: WR={metrics['combined_win_rate']:.1%}, RMSE={metrics['rmse']:.4f}, Avg VIF={metrics['avg_vif']:.2f}")
        
        # Test loai bo features co VIF > 10
        print("\nDang test loai bo features co VIF > 10...")
        high_vif_features = self.vif_data[self.vif_data['vif'] > 10]['feature'].tolist()
        test_features = [f for f in feature_cols if f not in high_vif_features]
        test_indices = [feature_to_idx[f] for f in test_features if f in feature_to_idx]
        
        if len(test_indices) > 0 and len(high_vif_features) > 0:
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices, epochs=50
            )
            metrics['test_type'] = 'remove_vif_gt10'
            metrics['removed_features'] = ','.join(high_vif_features)
            remaining_vif = self.vif_data[self.vif_data['feature'].isin(test_features)]
            metrics['avg_vif'] = remaining_vif['vif'].mean() if len(remaining_vif) > 0 else 0
            self.results.append(metrics)
            print(f"  Remove VIF>10: WR={metrics['combined_win_rate']:.1%}, RMSE={metrics['rmse']:.4f}, Avg VIF={metrics['avg_vif']:.2f}")
        
        # Tim best configuration
        results_df = pd.DataFrame(self.results)
        if len(results_df) > 0:
            # Tim best theo combined win rate va avg VIF
            results_df['score'] = results_df['combined_win_rate'] - (results_df['avg_vif'] / 100)
            best = results_df.loc[results_df['score'].idxmax()]
            
            print("\n" + "="*60)
            print("KET QUA TOI UU")
            print("="*60)
            print(f"Best Configuration: {best['test_type']}")
            print(f"  Combined Win Rate: {best['combined_win_rate']:.1%}")
            print(f"  Buy WR: {best['buy_win_rate']:.1%}, Sell WR: {best['sell_win_rate']:.1%}")
            print(f"  RMSE: {best['rmse']:.4f}")
            print(f"  Average VIF: {best['avg_vif']:.2f}")
            print(f"  Number of features: {best['num_features']}")
            if best['removed_features']:
                print(f"  Removed features: {best['removed_features']}")
        
        return results_df
    
    def save_results(self, results_df):
        """Luu ket qua"""
        results_dir = os.path.join(os.path.dirname(__file__), config.RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        
        save_path = os.path.join(results_dir, 'quick_vif_ablation_results.csv')
        results_df.to_csv(save_path, index=False)
        print(f"\nDa luu ket qua vao {save_path}")


def main():
    """Main workflow"""
    tester = QuickVIFAblation()
    results = tester.run_ablation()
    tester.save_results(results)
    
    print("\nHoan thanh quick VIF ablation test!")


if __name__ == "__main__":
    main()

