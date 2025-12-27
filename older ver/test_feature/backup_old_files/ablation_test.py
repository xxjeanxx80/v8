#!/usr/bin/env python3
"""
Ablation Testing Framework for NVDA LSTM Features
Tests impact of removing feature groups on model performance
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import our model classes
from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete, RegressionDataset, LSTMRegressor

class FeatureAblationTester:
    def __init__(self, data_dir="../data"):
        """Initialize with data directory"""
        self.data_dir = data_dir
        self.predictor = NVDA_MultiStock_Complete()
        self.results = {}
        self.feature_groups = {}
        
    def load_data(self):
        """Load and prepare data for testing"""
        print("📊 Loading multi-stock data...")
        df, feature_cols = self.predictor.load_multi_stock_data(self.data_dir)
        
        # Use the same split logic as main script
        (X_train, y_train_reg, y_train_cls,
         X_test, y_test_reg, y_test_cls) = self.predictor.split_data_transfer_learning(df, feature_cols)
        
        print(f"  Train: {X_train.shape[0]} sequences")
        print(f"  Test: {X_test.shape[0]} sequences (NVDA only)")
        
        return X_train, X_test, y_train_reg, y_test_reg, feature_cols
    
    def define_feature_groups(self, feature_cols):
        """Define feature groups for ablation testing"""
        groups = {
            'all': feature_cols,  # Full feature set
            'no_trend': [f for f in feature_cols if not any(x in f.lower() for x in ['sma', 'price_vs'])],
            'no_momentum': [f for f in feature_cols if not any(x in f.lower() for x in ['rsi', 'macd'])],
            'no_volatility': [f for f in feature_cols if not any(x in f.lower() for x in ['bb', 'atr', 'bollinger'])],
            'no_volume': [f for f in feature_cols if not any(x in f.lower() for x in ['volume', 'obv'])],
            'no_returns': [f for f in feature_cols if not any(x in f.lower() for x in ['return', 'change'])],
            'no_structure': [f for f in feature_cols if not any(x in f.lower() for x in ['spread', 'hl_', 'oc_'])],
            'reduced': [  # Core 14-16 features
                f for f in feature_cols if any(x in f.lower() for x in [
                    'price_vs_sma', 'rsi14', 'macd', 'atr', 'bb_bandwidth', 
                    'bb_percent', 'volume_ratio', 'obv', 'hl_spread_pct',
                    'daily_return', 'price_change', 'bb_squeeze'
                ])
            ]
        }
        
        # Filter out empty groups
        self.feature_groups = {k: v for k, v in groups.items() if len(v) > 0}
        
        print("\n📋 Feature Groups for Ablation:")
        for group, features in self.feature_groups.items():
            print(f"  {group:15s}: {len(features):2d} features")
        
        return self.feature_groups
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, feature_indices, epochs=50):
        """Train model with specific features and evaluate"""
        # Select features
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
        
        # Create model
        model = LSTMRegressor(input_size=len(feature_indices))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(epochs):
            # Simple training (no validation for speed)
            optimizer.zero_grad()
            outputs = model(torch.FloatTensor(X_train_scaled))
            loss = criterion(outputs, torch.FloatTensor(y_train_scaled))
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch:3d}: Loss = {loss.item():.6f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(torch.FloatTensor(X_test_scaled)).cpu().numpy()
            # Flatten if needed
            if y_pred_scaled.ndim > 1:
                y_pred_scaled = y_pred_scaled.flatten()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Trading metrics
        threshold = 0.02
        signals = np.zeros(len(y_pred))
        signals[y_pred > threshold] = 2  # BUY
        signals[y_pred < -threshold] = 0  # SELL
        
        buy_returns = y_test[signals == 2]
        sell_returns = y_test[signals == 0]
        
        buy_win_rate = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
        sell_win_rate = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
        coverage = np.sum((signals == 2) | (signals == 0)) / len(signals)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'coverage': coverage,
            'num_features': len(feature_indices)
        }
    
    def run_ablation_test(self, X_train, X_test, y_train, y_test, feature_cols):
        """Run complete ablation test across all feature groups"""
        print("\n🧪 Running Ablation Tests...")
        print("="*60)
        
        # Map feature names to indices
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        
        results = []
        
        for group_name, group_features in self.feature_groups.items():
            print(f"\n📊 Testing: {group_name.upper()}")
            print(f"  Features: {len(group_features)}")
            
            # Get feature indices
            feature_indices = [feature_to_idx[f] for f in group_features]
            
            # Train and evaluate
            metrics = self.train_and_evaluate(
                X_train, X_test, y_train, y_test, 
                feature_indices, epochs=30
            )
            
            metrics['group'] = group_name
            metrics['features'] = group_features
            results.append(metrics)
            
            print(f"  Results:")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    Buy Win Rate: {metrics['buy_win_rate']:.1%}")
            print(f"    Coverage: {metrics['coverage']:.1%}")
        
        # Store results
        self.results = pd.DataFrame(results)
        
        return self.results
    
    def statistical_significance_test(self):
        """Test statistical significance of performance differences"""
        print("\n📈 Statistical Significance Testing")
        print("="*60)
        
        # Compare full vs reduced feature set
        full_metrics = self.results[self.results['group'] == 'all'].iloc[0]
        reduced_metrics = self.results[self.results['group'] == 'reduced'].iloc[0]
        
        metrics_to_test = ['rmse', 'mae', 'buy_win_rate', 'coverage']
        
        print("\nFull vs Reduced Feature Set:")
        print(f"{'Metric':15s} {'Full':10s} {'Reduced':10s} {'Difference':12s} {'Significant?':12s}")
        print("-"*65)
        
        for metric in metrics_to_test:
            full_val = full_metrics[metric]
            reduced_val = reduced_metrics[metric]
            diff = reduced_val - full_val
            
            # For simplicity, we'll use percentage difference
            # In practice, you'd want multiple runs for statistical testing
            pct_diff = (diff / full_val) * 100
            
            # Simple heuristic for significance
            is_significant = abs(pct_diff) > 5  # 5% threshold
            
            print(f"{metric:15s} {full_val:10.4f} {reduced_val:10.4f} {pct_diff:12.2f}% {'Yes' if is_significant else 'No':12s}")
    
    def plot_results(self, save_path=None):
        """Create visualization of ablation test results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Ablation Test Results', fontsize=16)
        
        # RMSE
        ax = axes[0, 0]
        sns.barplot(data=self.results, x='group', y='rmse', ax=ax)
        ax.set_title('RMSE by Feature Group')
        ax.set_xlabel('Feature Group')
        ax.set_ylabel('RMSE')
        ax.tick_params(axis='x', rotation=45)
        
        # Buy Win Rate
        ax = axes[0, 1]
        sns.barplot(data=self.results, x='group', y='buy_win_rate', ax=ax)
        ax.set_title('Buy Win Rate by Feature Group')
        ax.set_xlabel('Feature Group')
        ax.set_ylabel('Win Rate')
        ax.tick_params(axis='x', rotation=45)
        
        # Coverage
        ax = axes[1, 0]
        sns.barplot(data=self.results, x='group', y='coverage', ax=ax)
        ax.set_title('Coverage by Feature Group')
        ax.set_xlabel('Feature Group')
        ax.set_ylabel('Coverage')
        ax.tick_params(axis='x', rotation=45)
        
        # Number of Features vs Performance
        ax = axes[1, 1]
        sns.scatterplot(data=self.results, x='num_features', y='rmse', s=100, ax=ax)
        ax.set_title('Number of Features vs RMSE')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('RMSE')
        
        # Add labels for points
        for _, row in self.results.iterrows():
            ax.annotate(row['group'], (row['num_features'], row['rmse']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved plots to {save_path}")
        else:
            plt.show()
    
    def generate_report(self):
        """Generate summary report"""
        print("\n📄 Feature Ablation Report")
        print("="*60)
        
        # Best performing configurations
        best_rmse = self.results.loc[self.results['rmse'].idxmin()]
        best_win_rate = self.results.loc[self.results['buy_win_rate'].idxmax()]
        
        print(f"\n🏆 Best RMSE: {best_rmse['group']} (RMSE = {best_rmse['rmse']:.4f})")
        print(f"   Features: {best_rmse['num_features']}")
        
        print(f"\n🎯 Best Win Rate: {best_win_rate['group']} (Win Rate = {best_win_rate['buy_win_rate']:.1%})")
        print(f"   Features: {best_win_rate['num_features']}")
        
        # Feature efficiency
        print(f"\n💡 Feature Efficiency Analysis:")
        for _, row in self.results.iterrows():
            efficiency = row['buy_win_rate'] / row['num_features'] * 100
            print(f"   {row['group']:15s}: {efficiency:.2f} win rate per feature")
        
        # Recommendations
        print(f"\n📋 Recommendations:")
        reduced = self.results[self.results['group'] == 'reduced'].iloc[0]
        full = self.results[self.results['group'] == 'all'].iloc[0]
        
        rmse_increase = (reduced['rmse'] - full['rmse']) / full['rmse'] * 100
        feature_reduction = (full['num_features'] - reduced['num_features']) / full['num_features'] * 100
        
        print(f"   • Reduced feature set: {feature_reduction:.1f}% fewer features")
        print(f"   • RMSE increase: {rmse_increase:.1f}%")
        
        if rmse_increase < 10:
            print(f"   ✅ RECOMMENDATION: Use reduced feature set (minimal performance loss)")
        else:
            print(f"   ⚠️  RECOMMENDATION: Keep full feature set (significant performance loss)")


def main():
    """Main ablation testing workflow"""
    print("="*60)
    print("🧪 FEATURE ABLATION TESTING")
    print("="*60)
    
    # Initialize tester
    tester = FeatureAblationTester()
    
    # Load data
    X_train, X_test, y_train, y_test, feature_cols = tester.load_data()
    
    # Define feature groups
    groups = tester.define_feature_groups(feature_cols)
    
    # Run ablation tests
    results = tester.run_ablation_test(X_train, X_test, y_train, y_test, feature_cols)
    
    # Statistical testing
    tester.statistical_significance_test()
    
    # Generate report
    tester.generate_report()
    
    # Visualizations
    print("\n📊 Generating visualizations...")
    tester.plot_results('ablation_results.png')
    
    # Save results
    results.to_csv('ablation_results.csv', index=False)
    print("\n💾 Results saved to ablation_results.csv")
    
    print("\n✅ Ablation testing complete!")


if __name__ == "__main__":
    main()
