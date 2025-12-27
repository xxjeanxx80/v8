#!/usr/bin/env python3
"""
Threshold Optimization Test
Tìm threshold tối ưu riêng cho BUY và SELL signals để tối đa hóa tổng win rate
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
import matplotlib.pyplot as plt
import seaborn as sns

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

class ThresholdOptimizer:
    def __init__(self, data_dir=None):
        """Khởi tạo với data directory"""
        self.data_dir = data_dir or config.DATA_DIR
        self.predictor = NVDA_MultiStock_Complete(
            sequence_length=config.SEQUENCE_LENGTH,
            horizon=config.HORIZON
        )
        self.results = []
        
    def load_data(self):
        """Load và chuẩn bị data cho testing"""
        print("Dang load du lieu multi-stock...")
        df, feature_cols = self.predictor.load_multi_stock_data(self.data_dir)
        
        # Su dung cung split logic nhu main script
        (X_train, y_train_reg, y_train_cls,
         X_test, y_test_reg, y_test_cls) = self.predictor.split_data_transfer_learning(df, feature_cols)
        
        print(f"  Train: {X_train.shape[0]} sequences")
        print(f"  Test: {X_test.shape[0]} sequences (NVDA only)")
        
        return X_train, X_test, y_train_reg, y_test_reg, feature_cols
    
    def train_model(self, X_train, X_test, y_train, y_test, feature_cols, epochs=None):
        """Train model mot lan va luu predictions"""
        epochs = epochs or config.EPOCHS
        
        # Scale data
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        
        # Tao model va chuyen sang GPU neu co
        model = LSTMRegressor(input_size=len(feature_cols))
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
            
            if config.VERBOSE and epoch % config.PRINT_EVERY_N_EPOCHS == 0:
                print(f"    Epoch {epoch:3d}: Loss = {loss.item():.6f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    if config.VERBOSE:
                        print(f"    Early stopping at epoch {epoch}")
                    break
        
        # Evaluate va lay predictions
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
            if y_pred_scaled.ndim > 1:
                y_pred_scaled = y_pred_scaled.flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred, y_test
    
    def evaluate_thresholds(self, y_pred, y_test, buy_percentile, sell_percentile):
        """Danh gia performance voi threshold cu the cho buy va sell"""
        # Tinh threshold rieng cho buy va sell
        buy_threshold = np.percentile(y_pred[y_pred > 0], buy_percentile) if np.any(y_pred > 0) else np.percentile(np.abs(y_pred), buy_percentile)
        sell_threshold = -np.percentile(-y_pred[y_pred < 0], 100 - sell_percentile) if np.any(y_pred < 0) else -np.percentile(np.abs(y_pred), 100 - sell_percentile)
        
        # Neu khong co du lieu de tinh percentile, su dung absolute percentile
        if np.isnan(buy_threshold) or buy_threshold <= 0:
            buy_threshold = np.percentile(np.abs(y_pred), buy_percentile)
        if np.isnan(sell_threshold) or sell_threshold >= 0:
            sell_threshold = -np.percentile(np.abs(y_pred), 100 - sell_percentile)
        
        # Tao signals
        signals = np.zeros(len(y_pred))
        signals[y_pred > buy_threshold] = 2  # BUY
        signals[y_pred < sell_threshold] = 0  # SELL
        # signals = 1 la NO_TRADE (mac dinh)
        
        # Tinh metrics
        buy_returns = y_test[signals == 2]
        sell_returns = y_test[signals == 0]
        
        buy_win_rate = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
        sell_win_rate = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
        combined_win_rate = (buy_win_rate + sell_win_rate) / 2 if (len(buy_returns) > 0 or len(sell_returns) > 0) else 0
        
        buy_coverage = len(buy_returns) / len(signals)
        sell_coverage = len(sell_returns) / len(signals)
        total_coverage = (len(buy_returns) + len(sell_returns)) / len(signals)
        
        # Weighted win rate
        weighted_win_rate = buy_win_rate * buy_coverage + sell_win_rate * sell_coverage
        
        # Sharpe-like metric
        sharpe_like = (buy_win_rate - 0.5) + (sell_win_rate - 0.5)
        
        return {
            'buy_percentile': buy_percentile,
            'sell_percentile': sell_percentile,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'combined_win_rate': combined_win_rate,
            'weighted_win_rate': weighted_win_rate,
            'sharpe_like': sharpe_like,
            'buy_coverage': buy_coverage,
            'sell_coverage': sell_coverage,
            'total_coverage': total_coverage,
            'num_buy_signals': len(buy_returns),
            'num_sell_signals': len(sell_returns)
        }
    
    def run_optimization(self):
        """Chay grid search tren cac threshold combinations"""
        print("="*60)
        print("THRESHOLD OPTIMIZATION TEST")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test, feature_cols = self.load_data()
        
        # Train model mot lan
        print("\nDang train model...")
        y_pred, y_test = self.train_model(X_train, X_test, y_train, y_test, feature_cols)
        
        print(f"\nPrediction range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
        print(f"Prediction mean: {np.mean(y_pred):.4f}, std: {np.std(y_pred):.4f}")
        
        # Grid search tren cac percentile combinations
        print("\nDang chay grid search tren cac threshold combinations...")
        print(f"Test {len(config.THRESHOLD_PERCENTILES)} x {len(config.THRESHOLD_PERCENTILES)} = {len(config.THRESHOLD_PERCENTILES)**2} combinations")
        
        results = []
        for buy_pct in config.THRESHOLD_PERCENTILES:
            for sell_pct in config.THRESHOLD_PERCENTILES:
                metrics = self.evaluate_thresholds(y_pred, y_test, buy_pct, sell_pct)
                results.append(metrics)
                
                if config.VERBOSE and (buy_pct == config.THRESHOLD_PERCENTILES[0] or 
                                      (buy_pct == config.THRESHOLD_PERCENTILES[-1] and sell_pct == config.THRESHOLD_PERCENTILES[-1])):
                    print(f"  Buy={buy_pct:2d}%, Sell={sell_pct:2d}%: "
                          f"Combined WR={metrics['combined_win_rate']:.1%}, "
                          f"Buy WR={metrics['buy_win_rate']:.1%}, "
                          f"Sell WR={metrics['sell_win_rate']:.1%}")
        
        self.results = pd.DataFrame(results)
        
        # Tim best configuration
        best_combined = self.results.loc[self.results['combined_win_rate'].idxmax()]
        best_weighted = self.results.loc[self.results['weighted_win_rate'].idxmax()]
        best_sharpe = self.results.loc[self.results['sharpe_like'].idxmax()]
        
        print("\n" + "="*60)
        print("KET QUA TOI UU")
        print("="*60)
        print(f"\nBest Combined Win Rate:")
        print(f"  Buy percentile: {best_combined['buy_percentile']:.0f}%")
        print(f"  Sell percentile: {best_combined['sell_percentile']:.0f}%")
        print(f"  Combined WR: {best_combined['combined_win_rate']:.1%}")
        print(f"  Buy WR: {best_combined['buy_win_rate']:.1%}, Sell WR: {best_combined['sell_win_rate']:.1%}")
        print(f"  Coverage: {best_combined['total_coverage']:.1%}")
        
        print(f"\nBest Weighted Win Rate:")
        print(f"  Buy percentile: {best_weighted['buy_percentile']:.0f}%")
        print(f"  Sell percentile: {best_weighted['sell_percentile']:.0f}%")
        print(f"  Weighted WR: {best_weighted['weighted_win_rate']:.1%}")
        
        print(f"\nBest Sharpe-like Metric:")
        print(f"  Buy percentile: {best_sharpe['buy_percentile']:.0f}%")
        print(f"  Sell percentile: {best_sharpe['sell_percentile']:.0f}%")
        print(f"  Sharpe-like: {best_sharpe['sharpe_like']:.3f}")
        
        return self.results, best_combined, best_weighted, best_sharpe
    
    def visualize_results(self, save_path=None):
        """Tao heatmap visualization cho threshold optimization"""
        if len(self.results) == 0:
            print("Chua co ket qua de visualize")
            return
        
        save_path = save_path or config.OUTPUT_FILES['threshold_heatmap']
        
        # Tao pivot table cho combined win rate
        pivot_combined = self.results.pivot(
            index='buy_percentile',
            columns='sell_percentile',
            values='combined_win_rate'
        )
        
        # Tao figure voi 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Combined win rate heatmap
        sns.heatmap(pivot_combined, annot=True, fmt='.1%', cmap=config.HEATMAP_CMAP,
                   center=0.7, ax=axes[0], cbar_kws={'label': 'Combined Win Rate'})
        axes[0].set_title('Combined Win Rate', fontsize=14)
        axes[0].set_xlabel('Sell Percentile')
        axes[0].set_ylabel('Buy Percentile')
        
        # Buy win rate heatmap
        pivot_buy = self.results.pivot(
            index='buy_percentile',
            columns='sell_percentile',
            values='buy_win_rate'
        )
        sns.heatmap(pivot_buy, annot=True, fmt='.1%', cmap=config.HEATMAP_CMAP,
                   center=0.75, ax=axes[1], cbar_kws={'label': 'Buy Win Rate'})
        axes[1].set_title('Buy Win Rate', fontsize=14)
        axes[1].set_xlabel('Sell Percentile')
        axes[1].set_ylabel('Buy Percentile')
        
        # Sell win rate heatmap
        pivot_sell = self.results.pivot(
            index='buy_percentile',
            columns='sell_percentile',
            values='sell_win_rate'
        )
        sns.heatmap(pivot_sell, annot=True, fmt='.1%', cmap=config.HEATMAP_CMAP,
                   center=0.6, ax=axes[2], cbar_kws={'label': 'Sell Win Rate'})
        axes[2].set_title('Sell Win Rate', fontsize=14)
        axes[2].set_xlabel('Sell Percentile')
        axes[2].set_ylabel('Buy Percentile')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"\nDa luu heatmap vao {save_path}")
        plt.close()
    
    def save_results(self, save_path=None):
        """Luu ket qua vao CSV"""
        if len(self.results) == 0:
            print("Chua co ket qua de luu")
            return
        
        save_path = save_path or config.OUTPUT_FILES['threshold_optimization']
        self.results.to_csv(save_path, index=False)
        print(f"Da luu ket qua vao {save_path}")


def main():
    """Main workflow"""
    optimizer = ThresholdOptimizer()
    
    # Chay optimization
    results, best_combined, best_weighted, best_sharpe = optimizer.run_optimization()
    
    # Visualize
    optimizer.visualize_results()
    
    # Luu ket qua
    optimizer.save_results()
    
    print("\nHoan thanh threshold optimization!")


if __name__ == "__main__":
    main()

