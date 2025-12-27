#!/usr/bin/env python3
"""
Optimized Feature Configuration Based on Ablation Testing
Creates a balanced feature set with 25 features for best performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v4_multistock'))
sys.path.append('..')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Import our model classes
from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete, RegressionDataset, LSTMRegressor

class OptimizedFeatureModel:
    def __init__(self, data_dir="../data"):
        """Initialize with optimized feature configuration"""
        self.data_dir = data_dir
        self.predictor = NVDA_MultiStock_Complete()
        
        # Optimized feature set based on ablation results
        # Start with no_trend (31 features) and remove highly correlated redundant ones
        self.optimized_features = [
            # Momentum (keep all - critical for performance)
            'rsi14',
            'macd',
            'macd_bullish',
            'macd_signal',
            'macd_hist',
            
            # Volatility (keep bb_percent and bb_bandwidth, remove redundant bb_upper/lower/middle)
            'atr',
            'bb_bandwidth',
            'bb_percent',
            
            # Volume (keep all - important for confirmation)
            'volume_ratio',
            'obv',
            'volume_sma20',
            
            # Returns (keep all - critical for prediction)
            'daily_return',
            'price_change',
            'return_3d',
            'return_5d',
            'return_10d',
            'return_20d',
            
            # Structure (keep hl_spread_pct, remove redundant hl_spread)
            'hl_spread_pct',
            'oc_spread',
            'oc_spread_pct',
            
            # Position indicators (keep all - useful for entry timing)
            'bb_squeeze',
            'rsi_overbought',
            'rsi_oversold',
            
            # Sector features (keep all - important for multi-stock context)
            'sox_beta',
            'sox_correlation'
        ]
        
        print(f"🎯 Optimized Feature Configuration: {len(self.optimized_features)} features")
        print(f"   Removed: price_vs_sma50, price_vs_sma200 (trend features)")
        print(f"   Removed: bb_upper, bb_lower, bb_middle (redundant volatility)")
        print(f"   Removed: hl_spread (redundant with hl_spread_pct)")
        
    def load_and_prepare_data(self):
        """Load data with optimized features"""
        print("\n📊 Loading data with optimized features...")
        
        # Load all data
        df, all_features = self.predictor.load_multi_stock_data(self.data_dir)
        
        # Filter to optimized features
        available_features = [f for f in self.optimized_features if f in all_features]
        missing_features = set(self.optimized_features) - set(available_features)
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        
        print(f"✅ Using {len(available_features)} optimized features")
        
        # Prepare train/test split
        (X_train, y_train_reg, y_train_cls,
         X_test, y_test_reg, y_test_cls) = self.predictor.split_data_transfer_learning(df, available_features)
        
        return X_train, X_test, y_train_reg, y_test_reg, available_features
    
    def train_optimized_model(self, X_train, X_test, y_train, y_test, feature_cols, epochs=100):
        """Train model with optimized feature configuration"""
        print(f"\n🔧 Training Optimized Model ({len(feature_cols)} features)...")
        
        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        
        # Create model
        model = LSTMRegressor(input_size=len(feature_cols))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Train with early stopping
        model.train()
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(torch.FloatTensor(X_train_scaled))
            loss = criterion(outputs, torch.FloatTensor(y_train_scaled))
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d}: Loss = {loss.item():.6f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(torch.FloatTensor(X_test_scaled)).cpu().numpy()
            if y_pred_scaled.ndim > 1:
                y_pred_scaled = y_pred_scaled.flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Trading metrics
        threshold = np.percentile(np.abs(y_pred), 75)
        signals = np.zeros(len(y_pred))
        signals[y_pred > threshold] = 2  # BUY
        signals[y_pred < -threshold] = 0  # SELL
        
        buy_returns = y_test[signals == 2]
        sell_returns = y_test[signals == 0]
        
        buy_win_rate = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
        sell_win_rate = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
        coverage = np.sum((signals == 2) | (signals == 0)) / len(signals)
        
        print(f"\n📊 Optimized Model Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Buy Win Rate: {buy_win_rate:.1%}")
        print(f"  Sell Win Rate: {sell_win_rate:.1%}")
        print(f"  Coverage: {coverage:.1%}")
        print(f"  Prediction Range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
        
        return {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'features': feature_cols,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'buy_win_rate': buy_win_rate,
                'sell_win_rate': sell_win_rate,
                'coverage': coverage
            }
        }
    
    def compare_with_baseline(self, optimized_results):
        """Compare optimized model with full feature baseline"""
        print("\n📈 Comparison with Baseline:")
        print("="*60)
        
        # Baseline results from previous tests
        baseline = {
            'features': 36,
            'rmse': 0.0411,
            'buy_win_rate': 0.758,
            'sell_win_rate': 0.647,
            'coverage': 1.0
        }
        
        optimized = {
            'features': len(optimized_results['features']),
            'rmse': optimized_results['metrics']['rmse'],
            'buy_win_rate': optimized_results['metrics']['buy_win_rate'],
            'sell_win_rate': optimized_results['metrics']['sell_win_rate'],
            'coverage': optimized_results['metrics']['coverage']
        }
        
        print(f"{'Metric':15s} {'Baseline':10s} {'Optimized':10s} {'Change':10s}")
        print("-"*50)
        
        for metric in ['features', 'rmse', 'buy_win_rate', 'sell_win_rate']:
            base_val = baseline[metric]
            opt_val = optimized[metric]
            
            if metric == 'features':
                change = (opt_val - base_val) / base_val * 100
                print(f"{metric:15s} {base_val:10d} {opt_val:10d} {change:+9.1f}%")
            elif metric in ['rmse']:
                change = (opt_val - base_val) / base_val * 100
                print(f"{metric:15s} {base_val:10.4f} {opt_val:10.4f} {change:+9.1f}%")
            else:
                change = (opt_val - base_val) * 100
                print(f"{metric:15s} {base_val:10.1%} {opt_val:10.1%} {change:+9.1%}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        feature_reduction = (baseline['features'] - optimized['features']) / baseline['features'] * 100
        rmse_change = (optimized['rmse'] - baseline['rmse']) / baseline['rmse'] * 100
        win_rate_change = (optimized['buy_win_rate'] - baseline['buy_win_rate']) * 100
        
        print(f"  • Feature reduction: {feature_reduction:.1f}%")
        print(f"  • RMSE change: {rmse_change:+.1f}%")
        print(f"  • Win rate change: {win_rate_change:+.1f}%")
        
        if feature_reduction > 20 and rmse_change < 5:
            print(f"  ✅ ADOPT optimized configuration (significant reduction, minimal performance loss)")
        elif win_rate_change > 5:
            print(f"  ✅ ADOPT optimized configuration (improved trading performance)")
        else:
            print(f"  ⚠️  Consider further optimization")
    
    def save_feature_config(self):
        """Save the optimized feature configuration"""
        config = {
            'features': self.optimized_features,
            'count': len(self.optimized_features),
            'rationale': [
                "Removed trend features (price_vs_sma) - improved win rate in ablation test",
                "Removed redundant Bollinger Bands (upper/lower/middle) - kept percent and bandwidth",
                "Removed redundant spread (hl_spread) - kept percentage version",
                "Kept all momentum indicators - critical for performance",
                "Kept all return features - essential for prediction",
                "Kept volume features - important for confirmation",
                "Kept sector features - important for multi-stock context"
            ]
        }
        
        df_config = pd.DataFrame([config])
        df_config.to_csv('optimized_feature_config.csv', index=False)
        print(f"\n💾 Saved feature configuration to optimized_feature_config.csv")


def main():
    """Main optimized model workflow"""
    print("="*60)
    print("🎯 OPTIMIZED FEATURE MODEL")
    print("="*60)
    
    # Initialize
    optimizer = OptimizedFeatureModel()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_cols = optimizer.load_and_prepare_data()
    
    # Train optimized model
    results = optimizer.train_optimized_model(X_train, X_test, y_train, y_test, feature_cols)
    
    # Compare with baseline
    optimizer.compare_with_baseline(results)
    
    # Save configuration
    optimizer.save_feature_config()
    
    print("\n✅ Optimization complete!")


if __name__ == "__main__":
    main()
