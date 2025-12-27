#!/usr/bin/env python3
"""
Feature Combination Testing
Test cac to hop feature chi tiet dua tren ket qua ablation de tim combination tot nhat
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
import torch

# Hien thi thong tin device
if config.DEVICE:
    print(f"Su dung device: {config.DEVICE}")
    if config.USE_GPU:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Canh bao: Dang su dung CPU. De su dung GPU, chay: python install_pytorch_cuda.py")

class FeatureCombinationTester:
    def __init__(self, data_dir=None):
        """Khoi tao voi data directory"""
        self.data_dir = data_dir or config.DATA_DIR
        self.predictor = NVDA_MultiStock_Complete(
            sequence_length=config.SEQUENCE_LENGTH,
            horizon=config.HORIZON
        )
        self.results = []
        self.feature_importance = {}
        
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
    
    def get_feature_groups(self, feature_cols):
        """Lay cac nhom feature tu config"""
        groups = {}
        for group_name, group_features in config.FEATURE_GROUPS.items():
            existing_features = [f for f in group_features if f in feature_cols]
            if existing_features:
                groups[group_name] = existing_features
        return groups
    
    def get_no_trend_features(self, feature_cols):
        """Lay feature set no_trend (31 features, buy_win_rate 87.88%)"""
        no_trend = [f for f in feature_cols if not any(x in f.lower() for x in ['sma', 'price_vs'])]
        return no_trend
    
    def test_feature_addition(self, X_train, X_test, y_train, y_test, feature_cols, 
                             base_features, candidate_features, buy_threshold_pct=75, sell_threshold_pct=75):
        """Test them tung feature vao base set"""
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        base_indices = [feature_to_idx[f] for f in base_features if f in feature_to_idx]
        
        results = []
        for feature in candidate_features:
            if feature not in feature_to_idx:
                continue
            
            test_features = base_features + [feature]
            test_indices = base_indices + [feature_to_idx[feature]]
            
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices,
                buy_threshold_pct, sell_threshold_pct
            )
            
            metrics['feature_added'] = feature
            metrics['base_features'] = ','.join(base_features)
            metrics['test_features'] = ','.join(test_features)
            results.append(metrics)
            
            # Luu feature importance
            improvement = metrics['combined_win_rate'] - self.get_baseline_win_rate(base_features, feature_cols)
            self.feature_importance[feature] = self.feature_importance.get(feature, 0) + improvement
        
        return results
    
    def test_feature_removal(self, X_train, X_test, y_train, y_test, feature_cols,
                            base_features, buy_threshold_pct=75, sell_threshold_pct=75):
        """Test loai bo tung feature tu base set"""
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        base_indices = [feature_to_idx[f] for f in base_features if f in feature_to_idx]
        
        # Baseline performance
        baseline_metrics = self.train_and_evaluate(
            X_train, X_test, y_train, y_test, base_indices,
            buy_threshold_pct, sell_threshold_pct
        )
        baseline_wr = baseline_metrics['combined_win_rate']
        
        results = []
        for i, feature in enumerate(base_features):
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
            
            metrics['feature_removed'] = feature
            metrics['base_features'] = ','.join(base_features)
            metrics['test_features'] = ','.join(test_features)
            metrics['impact'] = baseline_wr - metrics['combined_win_rate']
            results.append(metrics)
            
            # Luu feature importance (negative impact khi remove)
            self.feature_importance[feature] = self.feature_importance.get(feature, 0) + metrics['impact']
        
        return results
    
    def get_baseline_win_rate(self, features, feature_cols):
        """Lay baseline win rate (tam thoi tra ve 0, se duoc cap nhat sau)"""
        return 0.0
    
    def test_momentum_combinations(self, X_train, X_test, y_train, y_test, feature_cols, 
                                   base_features, buy_threshold_pct=75, sell_threshold_pct=75):
        """Test cac to hop momentum features"""
        groups = self.get_feature_groups(feature_cols)
        momentum_features = groups.get('momentum', [])
        
        if not momentum_features:
            return []
        
        results = []
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        base_indices = [feature_to_idx[f] for f in base_features if f in feature_to_idx]
        
        # Test tung momentum feature
        for feature in momentum_features:
            if feature not in feature_to_idx:
                continue
            
            test_features = base_features + [feature] if feature not in base_features else base_features
            test_indices = base_indices + ([feature_to_idx[feature]] if feature not in base_features else [])
            
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices,
                buy_threshold_pct, sell_threshold_pct
            )
            
            metrics['momentum_feature'] = feature
            metrics['test_features'] = ','.join(test_features)
            results.append(metrics)
        
        return results
    
    def test_volatility_combinations(self, X_train, X_test, y_train, y_test, feature_cols,
                                    base_features, buy_threshold_pct=75, sell_threshold_pct=75):
        """Test cac to hop volatility features"""
        groups = self.get_feature_groups(feature_cols)
        volatility_features = groups.get('volatility', [])
        
        if not volatility_features:
            return []
        
        results = []
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        base_indices = [feature_to_idx[f] for f in base_features if f in feature_to_idx]
        
        # Test cac to hop volatility
        for feature in volatility_features:
            if feature not in feature_to_idx:
                continue
            
            test_features = base_features + [feature] if feature not in base_features else base_features
            test_indices = base_indices + ([feature_to_idx[feature]] if feature not in base_features else [])
            
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices,
                buy_threshold_pct, sell_threshold_pct
            )
            
            metrics['volatility_feature'] = feature
            metrics['test_features'] = ','.join(test_features)
            results.append(metrics)
        
        return results
    
    def test_sector_features(self, X_train, X_test, y_train, y_test, feature_cols,
                            base_features, buy_threshold_pct=75, sell_threshold_pct=75):
        """Test impact cua sector features"""
        groups = self.get_feature_groups(feature_cols)
        sector_features = groups.get('sector', [])
        
        if not sector_features:
            return []
        
        results = []
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        base_indices = [feature_to_idx[f] for f in base_features if f in feature_to_idx]
        
        # Test voi va khong co sector features
        # Khong co sector
        metrics_no_sector = self.train_and_evaluate(
            X_train, X_test, y_train, y_test, base_indices,
            buy_threshold_pct, sell_threshold_pct
        )
        metrics_no_sector['has_sector'] = False
        metrics_no_sector['test_features'] = ','.join(base_features)
        results.append(metrics_no_sector)
        
        # Co sector
        sector_in_base = [f for f in sector_features if f in base_features]
        if sector_in_base:
            metrics_with_sector = metrics_no_sector.copy()
            metrics_with_sector['has_sector'] = True
            results.append(metrics_with_sector)
        else:
            # Them sector features
            test_features = base_features + [f for f in sector_features if f in feature_to_idx]
            test_indices = base_indices + [feature_to_idx[f] for f in sector_features if f in feature_to_idx]
            
            metrics_with_sector = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, test_indices,
                buy_threshold_pct, sell_threshold_pct
            )
            metrics_with_sector['has_sector'] = True
            metrics_with_sector['test_features'] = ','.join(test_features)
            results.append(metrics_with_sector)
        
        return results
    
    def forward_selection(self, X_train, X_test, y_train, y_test, feature_cols,
                         buy_threshold_pct=75, sell_threshold_pct=75):
        """Forward selection: bat dau tu core features, them tung feature mot"""
        print("\nDang chay Forward Selection...")
        
        # Bat dau tu core features
        current_features = [f for f in config.CORE_FEATURES if f in feature_cols]
        all_candidates = [f for f in feature_cols if f not in current_features]
        
        results = []
        best_wr = 0
        
        while len(all_candidates) > 0:
            best_feature = None
            best_metrics = None
            
            # Test them tung candidate
            for candidate in all_candidates[:20]:  # Gioi han de tranh qua lau
                test_features = current_features + [candidate]
                feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
                test_indices = [feature_to_idx[f] for f in test_features]
                
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices,
                    buy_threshold_pct, sell_threshold_pct
                )
                
                metrics['selection_step'] = len(current_features)
                metrics['feature_added'] = candidate
                metrics['test_features'] = ','.join(test_features)
                results.append(metrics)
                
                if metrics['combined_win_rate'] > best_wr:
                    best_wr = metrics['combined_win_rate']
                    best_feature = candidate
                    best_metrics = metrics
            
            if best_feature:
                current_features.append(best_feature)
                all_candidates.remove(best_feature)
                print(f"  Step {len(current_features)}: Them {best_feature}, WR={best_wr:.1%}")
            else:
                break
        
        return results
    
    def backward_elimination(self, X_train, X_test, y_train, y_test, feature_cols,
                           buy_threshold_pct=75, sell_threshold_pct=75):
        """Backward elimination: bat dau tu no_trend, loai bo tung feature mot"""
        print("\nDang chay Backward Elimination...")
        
        # Bat dau tu no_trend
        current_features = self.get_no_trend_features(feature_cols)
        results = []
        
        while len(current_features) > len(config.CORE_FEATURES):
            worst_feature = None
            best_wr = 0
            
            # Test loai bo tung feature
            for feature in current_features:
                test_features = [f for f in current_features if f != feature]
                feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
                test_indices = [feature_to_idx[f] for f in test_features]
                
                metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test, test_indices,
                    buy_threshold_pct, sell_threshold_pct
                )
                
                metrics['elimination_step'] = len(current_features)
                metrics['feature_removed'] = feature
                metrics['test_features'] = ','.join(test_features)
                results.append(metrics)
                
                if metrics['combined_win_rate'] > best_wr:
                    best_wr = metrics['combined_win_rate']
                    worst_feature = feature
            
            if worst_feature:
                current_features.remove(worst_feature)
                print(f"  Step {len(current_features)}: Loai bo {worst_feature}, WR={best_wr:.1%}")
            else:
                break
        
        return results
    
    def run_comprehensive_test(self, buy_threshold_pct=75, sell_threshold_pct=75):
        """Chay tat ca cac test"""
        print("="*60)
        print("FEATURE COMBINATION TESTING")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test, feature_cols = self.load_data()
        
        # Lay no_trend features (baseline tot nhat)
        no_trend_features = self.get_no_trend_features(feature_cols)
        print(f"\nBaseline: no_trend features ({len(no_trend_features)} features)")
        
        all_results = []
        
        # Test forward selection
        forward_results = self.forward_selection(
            X_train, X_test, y_train, y_test, feature_cols,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(forward_results)
        
        # Test backward elimination
        backward_results = self.backward_elimination(
            X_train, X_test, y_train, y_test, feature_cols,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(backward_results)
        
        # Test momentum combinations
        momentum_results = self.test_momentum_combinations(
            X_train, X_test, y_train, y_test, feature_cols, no_trend_features,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(momentum_results)
        
        # Test volatility combinations
        volatility_results = self.test_volatility_combinations(
            X_train, X_test, y_train, y_test, feature_cols, no_trend_features,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(volatility_results)
        
        # Test sector features
        sector_results = self.test_sector_features(
            X_train, X_test, y_train, y_test, feature_cols, no_trend_features,
            buy_threshold_pct, sell_threshold_pct
        )
        all_results.extend(sector_results)
        
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
            if 'test_features' in best:
                print(f"Features: {best['test_features'][:100]}...")
        
        # Tao feature importance ranking
        self.create_feature_importance_ranking()
        
        return self.results
    
    def create_feature_importance_ranking(self):
        """Tao ranking cac feature theo importance"""
        if not self.feature_importance:
            return
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in self.feature_importance.items()
        ])
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        save_path = config.OUTPUT_FILES['feature_importance']
        importance_df.to_csv(save_path, index=False)
        print(f"\nDa luu feature importance ranking vao {save_path}")
    
    def save_results(self, save_path=None):
        """Luu ket qua"""
        if len(self.results) == 0:
            print("Chua co ket qua de luu")
            return
        
        save_path = save_path or config.OUTPUT_FILES['feature_combination']
        self.results.to_csv(save_path, index=False)
        print(f"Da luu ket qua vao {save_path}")


def main():
    """Main workflow"""
    tester = FeatureCombinationTester()
    
    # Chay comprehensive test
    results = tester.run_comprehensive_test()
    
    # Luu ket qua
    tester.save_results()
    
    print("\nHoan thanh feature combination testing!")


if __name__ == "__main__":
    main()

