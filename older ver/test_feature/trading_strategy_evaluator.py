#!/usr/bin/env python3
"""
Trading Strategy Evaluator
Danh gia chien luoc trading tong hop voi cac metrics thuc te
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
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

class TradingStrategyEvaluator:
    def __init__(self, data_dir=None):
        """Khoi tao voi data directory"""
        self.data_dir = data_dir or config.DATA_DIR
        self.predictor = NVDA_MultiStock_Complete(
            sequence_length=config.SEQUENCE_LENGTH,
            horizon=config.HORIZON
        )
        self.results = []
        
    def load_data(self):
        """Load va chuan bi data"""
        print("Dang load du lieu multi-stock...")
        df, feature_cols = self.predictor.load_multi_stock_data(self.data_dir)
        
        (X_train, y_train_reg, y_train_cls,
         X_test, y_test_reg, y_test_cls) = self.predictor.split_data_transfer_learning(df, feature_cols)
        
        print(f"  Train: {X_train.shape[0]} sequences")
        print(f"  Test: {X_test.shape[0]} sequences (NVDA only)")
        
        return X_train, X_test, y_train_reg, y_test_reg, feature_cols, df
    
    def train_model(self, X_train, X_test, y_train, y_test, feature_indices, epochs=None):
        """Train model va tra ve predictions"""
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
        
        return y_pred, y_test
    
    def generate_signals(self, y_pred, buy_threshold_pct=75, sell_threshold_pct=75):
        """Tao signals tu predictions"""
        buy_threshold = np.percentile(y_pred[y_pred > 0], buy_threshold_pct) if np.any(y_pred > 0) else np.percentile(np.abs(y_pred), buy_threshold_pct)
        sell_threshold = -np.percentile(-y_pred[y_pred < 0], 100 - sell_threshold_pct) if np.any(y_pred < 0) else -np.percentile(np.abs(y_pred), 100 - sell_threshold_pct)
        
        if np.isnan(buy_threshold) or buy_threshold <= 0:
            buy_threshold = np.percentile(np.abs(y_pred), buy_threshold_pct)
        if np.isnan(sell_threshold) or sell_threshold >= 0:
            sell_threshold = -np.percentile(np.abs(y_pred), 100 - sell_threshold_pct)
        
        signals = np.ones(len(y_pred))  # Mac dinh la NO_TRADE
        signals[y_pred > buy_threshold] = 2  # BUY
        signals[y_pred < sell_threshold] = 0  # SELL
        
        return signals, buy_threshold, sell_threshold
    
    def calculate_trading_metrics(self, signals, y_true, y_pred):
        """Tinh cac metrics trading"""
        # Basic win rates
        buy_returns = y_true[signals == 2]
        sell_returns = y_true[signals == 0]
        
        buy_win_rate = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
        sell_win_rate = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
        combined_win_rate = (buy_win_rate + sell_win_rate) / 2 if (len(buy_returns) > 0 or len(sell_returns) > 0) else 0
        
        # Total return: Tinh tong loi nhuan neu follow signals
        total_return = 0.0
        cumulative_returns = []
        position = 0  # 0: no position, 1: long, -1: short
        
        for i in range(len(signals)):
            if signals[i] == 2 and position != 1:  # BUY signal
                position = 1
            elif signals[i] == 0 and position != -1:  # SELL signal
                position = -1
            elif signals[i] == 1:  # NO_TRADE
                position = 0
            
            # Tinh return cho ngay nay
            if position == 1:  # Long position
                daily_return = y_true[i]
            elif position == -1:  # Short position
                daily_return = -y_true[i]
            else:
                daily_return = 0
            
            total_return += daily_return
            cumulative_returns.append(total_return)
        
        # Sharpe ratio
        returns_array = np.array(cumulative_returns)
        if len(returns_array) > 1:
            returns_diff = np.diff(returns_array)
            sharpe_ratio = np.mean(returns_diff) / (np.std(returns_diff) + 1e-6) * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        if len(cumulative_returns) > 0:
            cumulative_array = np.array(cumulative_returns)
            running_max = np.maximum.accumulate(cumulative_array)
            drawdown = cumulative_array - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        # Win rate by signal strength
        buy_strengths = y_pred[signals == 2]
        sell_strengths = -y_pred[signals == 0]
        
        if len(buy_strengths) > 0:
            buy_median = np.median(buy_strengths)
            strong_buy = buy_returns[y_pred[signals == 2] > buy_median]
            weak_buy = buy_returns[y_pred[signals == 2] <= buy_median]
            strong_buy_wr = np.mean(strong_buy > 0) if len(strong_buy) > 0 else 0
            weak_buy_wr = np.mean(weak_buy > 0) if len(weak_buy) > 0 else 0
        else:
            strong_buy_wr = 0
            weak_buy_wr = 0
        
        if len(sell_strengths) > 0:
            sell_median = np.median(sell_strengths)
            strong_sell = sell_returns[-y_pred[signals == 0] > sell_median]
            weak_sell = sell_returns[-y_pred[signals == 0] <= sell_median]
            strong_sell_wr = np.mean(strong_sell < 0) if len(strong_sell) > 0 else 0
            weak_sell_wr = np.mean(weak_sell < 0) if len(weak_sell) > 0 else 0
        else:
            strong_sell_wr = 0
            weak_sell_wr = 0
        
        # Confusion matrix
        true_signals = np.zeros(len(y_true))
        true_signals[y_true > 0.02] = 2  # BUY
        true_signals[y_true < -0.02] = 0  # SELL
        # true_signals = 1 la NO_TRADE (mac dinh)
        
        cm = confusion_matrix(true_signals, signals, labels=[0, 1, 2])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1] + cm[0, 2], cm[1, 0] + cm[2, 0], cm[1, 1] + cm[2, 2]
        
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'combined_win_rate': combined_win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'strong_buy_wr': strong_buy_wr,
            'weak_buy_wr': weak_buy_wr,
            'strong_sell_wr': strong_sell_wr,
            'weak_sell_wr': weak_sell_wr,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'num_buy_signals': len(buy_returns),
            'num_sell_signals': len(sell_returns),
            'coverage': (len(buy_returns) + len(sell_returns)) / len(signals)
        }
    
    def compare_with_baseline(self, signals, y_true):
        """So sanh voi baseline strategies"""
        # Buy-and-hold
        buy_hold_return = np.sum(y_true)
        
        # Random signals (50% buy, 50% sell)
        np.random.seed(42)
        random_signals = np.random.choice([0, 1, 2], size=len(signals), p=[0.25, 0.5, 0.25])
        random_buy_returns = y_true[random_signals == 2]
        random_sell_returns = y_true[random_signals == 0]
        random_total_return = np.sum(random_buy_returns) - np.sum(random_sell_returns)
        
        # Strategy return
        strategy_return = 0.0
        position = 0
        for i in range(len(signals)):
            if signals[i] == 2 and position != 1:
                position = 1
            elif signals[i] == 0 and position != -1:
                position = -1
            elif signals[i] == 1:
                position = 0
            
            if position == 1:
                strategy_return += y_true[i]
            elif position == -1:
                strategy_return -= y_true[i]
        
        return {
            'buy_hold_return': buy_hold_return,
            'random_return': random_total_return,
            'strategy_return': strategy_return,
            'vs_buy_hold': strategy_return - buy_hold_return,
            'vs_random': strategy_return - random_total_return
        }
    
    def walk_forward_analysis(self, X_train, X_test, y_train, y_test, feature_indices,
                             buy_threshold_pct=75, sell_threshold_pct=75, n_periods=3):
        """Walk-forward analysis: test tren cac period khac nhau"""
        print("\nDang chay Walk-Forward Analysis...")
        
        period_size = len(y_test) // n_periods
        results = []
        
        for period in range(n_periods):
            start_idx = period * period_size
            end_idx = (period + 1) * period_size if period < n_periods - 1 else len(y_test)
            
            X_test_period = X_test[start_idx:end_idx]
            y_test_period = y_test[start_idx:end_idx]
            
            # Train model (co the retrain hoac su dung model da train)
            y_pred_period, _ = self.train_model(X_train, X_test_period, y_train, y_test_period, feature_indices)
            
            # Generate signals
            signals_period, _, _ = self.generate_signals(y_pred_period, buy_threshold_pct, sell_threshold_pct)
            
            # Calculate metrics
            metrics = self.calculate_trading_metrics(signals_period, y_test_period, y_pred_period)
            metrics['period'] = period
            metrics['period_start'] = start_idx
            metrics['period_end'] = end_idx
            results.append(metrics)
            
            if config.VERBOSE:
                print(f"  Period {period + 1}: WR = {metrics['combined_win_rate']:.1%}, "
                      f"Return = {metrics['total_return']:.4f}")
        
        return results
    
    def evaluate_strategy(self, feature_indices, feature_names, 
                        buy_threshold_pct=75, sell_threshold_pct=75):
        """Danh gia chien luoc trading"""
        print("="*60)
        print("TRADING STRATEGY EVALUATION")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test, feature_cols, df = self.load_data()
        
        print(f"\nDang train model voi {len(feature_indices)} features...")
        y_pred, y_test = self.train_model(X_train, X_test, y_train, y_test, feature_indices)
        
        # Generate signals
        signals, buy_threshold, sell_threshold = self.generate_signals(
            y_pred, buy_threshold_pct, sell_threshold_pct
        )
        
        print(f"\nThresholds: Buy = {buy_threshold:.4f}, Sell = {sell_threshold:.4f}")
        print(f"Signal distribution: BUY={np.sum(signals==2)}, SELL={np.sum(signals==0)}, NO_TRADE={np.sum(signals==1)}")
        
        # Calculate trading metrics
        metrics = self.calculate_trading_metrics(signals, y_test, y_pred)
        
        # Compare with baseline
        baseline_comparison = self.compare_with_baseline(signals, y_test)
        metrics.update(baseline_comparison)
        
        # Walk-forward analysis
        wf_results = self.walk_forward_analysis(
            X_train, X_test, y_train, y_test, feature_indices,
            buy_threshold_pct, sell_threshold_pct
        )
        
        # Print results
        print("\n" + "="*60)
        print("KET QUA DANH GIA")
        print("="*60)
        print(f"\nWin Rates:")
        print(f"  Buy Win Rate: {metrics['buy_win_rate']:.1%}")
        print(f"  Sell Win Rate: {metrics['sell_win_rate']:.1%}")
        print(f"  Combined Win Rate: {metrics['combined_win_rate']:.1%}")
        
        print(f"\nReturns:")
        print(f"  Total Return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
        
        print(f"\nComparison:")
        print(f"  vs Buy-and-Hold: {metrics['vs_buy_hold']:.4f} ({metrics['vs_buy_hold']*100:.2f}%)")
        print(f"  vs Random: {metrics['vs_random']:.4f} ({metrics['vs_random']*100:.2f}%)")
        
        print(f"\nSignal Strength Analysis:")
        print(f"  Strong Buy WR: {metrics['strong_buy_wr']:.1%}")
        print(f"  Weak Buy WR: {metrics['weak_buy_wr']:.1%}")
        print(f"  Strong Sell WR: {metrics['strong_sell_wr']:.1%}")
        print(f"  Weak Sell WR: {metrics['weak_sell_wr']:.1%}")
        
        print(f"\nError Rates:")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.1%}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.1%}")
        
        # Add metadata
        metrics['feature_count'] = len(feature_indices)
        metrics['feature_names'] = ','.join(feature_names)
        metrics['buy_threshold_pct'] = buy_threshold_pct
        metrics['sell_threshold_pct'] = sell_threshold_pct
        
        # Store results
        self.results = {
            'main_metrics': metrics,
            'walk_forward': wf_results
        }
        
        return metrics, wf_results
    
    def visualize_performance(self, signals, y_true, y_pred, save_path=None):
        """Visualize performance chart"""
        save_path = save_path or config.OUTPUT_FILES['trading_performance']
        
        # Cumulative returns
        cumulative_returns = []
        position = 0
        total_return = 0.0
        
        for i in range(len(signals)):
            if signals[i] == 2 and position != 1:
                position = 1
            elif signals[i] == 0 and position != -1:
                position = -1
            elif signals[i] == 1:
                position = 0
            
            if position == 1:
                total_return += y_true[i]
            elif position == -1:
                total_return -= y_true[i]
            
            cumulative_returns.append(total_return)
        
        # Buy-and-hold
        buy_hold_returns = np.cumsum(y_true)
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=config.FIGURE_SIZE)
        
        # Cumulative returns
        axes[0].plot(cumulative_returns, label='Strategy', linewidth=2)
        axes[0].plot(buy_hold_returns, label='Buy-and-Hold', linewidth=2, alpha=0.7)
        axes[0].set_title('Cumulative Returns', fontsize=14)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Signals
        axes[1].scatter(range(len(y_pred)), y_pred, c=signals, cmap='RdYlGn', alpha=0.6, s=20)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_title('Predictions and Signals', fontsize=14)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Predicted Return')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"\nDa luu performance chart vao {save_path}")
        plt.close()
    
    def save_report(self, save_path=None):
        """Luu bao cao"""
        if not self.results:
            print("Chua co ket qua de luu")
            return
        
        save_path = save_path or config.OUTPUT_FILES['trading_strategy']
        
        # Combine main metrics and walk-forward results
        report_data = [self.results['main_metrics']]
        for wf in self.results['walk_forward']:
            wf_copy = wf.copy()
            wf_copy['period_type'] = 'walk_forward'
            report_data.append(wf_copy)
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(save_path, index=False)
        print(f"Da luu bao cao vao {save_path}")


def main():
    """Main workflow"""
    evaluator = TradingStrategyEvaluator()
    
    # Load data de lay feature indices
    X_train, X_test, y_train, y_test, feature_cols, df = evaluator.load_data()
    
    # Su dung no_trend features (baseline tot nhat)
    no_trend_features = [f for f in feature_cols if not any(x in f.lower() for x in ['sma', 'price_vs'])]
    feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
    feature_indices = [feature_to_idx[f] for f in no_trend_features if f in feature_to_idx]
    
    # Danh gia chien luoc
    metrics, wf_results = evaluator.evaluate_strategy(
        feature_indices, no_trend_features,
        buy_threshold_pct=75, sell_threshold_pct=75
    )
    
    # Generate signals de visualize
    y_pred, _ = evaluator.train_model(X_train, X_test, y_train, y_test, feature_indices)
    signals, _, _ = evaluator.generate_signals(y_pred)
    
    # Visualize
    evaluator.visualize_performance(signals, y_test, y_pred)
    
    # Luu bao cao
    evaluator.save_report()
    
    print("\nHoan thanh trading strategy evaluation!")


if __name__ == "__main__":
    main()

