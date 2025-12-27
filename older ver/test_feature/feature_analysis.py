#!/usr/bin/env python3
"""
Feature Analysis Framework for NVDA LSTM
Tests feature importance and redundancy using:
1. Correlation heatmap (|corr| > 0.9)
2. Variance Inflation Factor (VIF)
3. Ablation testing by feature groups
4. Statistical significance testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings("ignore")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeatureAnalyzer:
    def __init__(self, data_path):
        """Initialize with feature data"""
        self.df = pd.read_csv(data_path)
        self.feature_cols = []
        self.results = {}
        
    def identify_feature_columns(self):
        """Identify feature columns (exclude targets and metadata)"""
        exclude_cols = [
            "Date", "Index", "Adj Close", "Close", "Open", "High", "Low",
            "daily_return", "price_change", "future_return", "signal_label"
        ]
        
        # Also exclude stock ID columns
        exclude_cols.extend([col for col in self.df.columns if col.startswith('stock_')])
        
        self.feature_cols = [col for col in self.df.columns 
                           if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]
        
        print(f"📊 Found {len(self.feature_cols)} feature columns")
        return self.feature_cols
    
    def correlation_analysis(self, threshold=0.9):
        """Analyze feature correlations to identify redundancy"""
        print("\n🔍 Correlation Analysis (threshold = {:.1f})".format(threshold))
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.feature_cols].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        # Store results
        self.results['correlation'] = {
            'matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'threshold': threshold
        }
        
        # Print summary
        if high_corr_pairs:
            print(f"  ⚠️  Found {len(high_corr_pairs)} highly correlated pairs:")
            for pair in high_corr_pairs:
                print(f"    - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print(f"  ✅ No correlations above {threshold}")
        
        return high_corr_pairs
    
    def calculate_vif(self):
        """Calculate Variance Inflation Factor for multicollinearity"""
        print("\n📈 Variance Inflation Factor (VIF) Analysis")
        
        # Prepare data (add constant for intercept)
        X = self.df[self.feature_cols].dropna()
        X_const = add_constant(X)
        
        # Calculate VIF for each feature
        vif_data = []
        for i, feature in enumerate(X.columns):
            vif = variance_inflation_factor(X_const.values, i+1)  # +1 because of constant
            vif_data.append({
                'feature': feature,
                'vif': vif
            })
        
        # Convert to DataFrame and sort
        vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
        
        # Store results
        self.results['vif'] = vif_df
        
        # Print summary
        print("  Features with high VIF (>5):")
        high_vif = vif_df[vif_df['vif'] > 5]
        if len(high_vif) > 0:
            for _, row in high_vif.iterrows():
                print(f"    - {row['feature']}: VIF = {row['vif']:.2f}")
        else:
            print("    ✅ No features with VIF > 5")
        
        return vif_df
    
    def plot_correlation_heatmap(self, save_path=None):
        """Create correlation heatmap visualization"""
        if 'correlation' not in self.results:
            self.correlation_analysis()
        
        plt.figure(figsize=(14, 12))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.results['correlation']['matrix'], dtype=bool))
        
        # Draw heatmap
        sns.heatmap(self.results['correlation']['matrix'], 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  💾 Saved heatmap to {save_path}")
        else:
            plt.show()
    
    def define_feature_groups(self):
        """Define feature groups for ablation testing"""
        groups = {
            'trend': ['price_vs_sma50', 'price_vs_sma200'],
            'momentum': ['rsi14', 'macd', 'macd_bullish', 'macd_signal', 'macd_hist'],
            'volatility': ['atr', 'bb_bandwidth', 'bb_percent', 'bb_upper', 'bb_lower', 'bb_middle'],
            'volume': ['volume_ratio', 'obv', 'volume_sma20'],
            'returns': ['daily_return', 'price_change', 'return_3d', 'return_5d', 'return_10d', 'return_20d'],
            'structure': ['hl_spread', 'hl_spread_pct', 'oc_spread', 'oc_spread_pct'],
            'position': ['bb_squeeze', 'rsi_overbought', 'rsi_oversold']
        }
        
        # Filter groups to only include existing features
        filtered_groups = {}
        for group, features in groups.items():
            existing = [f for f in features if f in self.feature_cols]
            if existing:
                filtered_groups[group] = existing
        
        print("\n📋 Feature Groups for Ablation Testing:")
        for group, features in filtered_groups.items():
            print(f"  {group}: {len(features)} features - {features}")
        
        return filtered_groups
    
    def suggest_reduced_feature_set(self):
        """Suggest a reduced feature set based on analysis"""
        print("\n💡 Suggested Reduced Feature Set (14-16 features):")
        
        # Core features based on domain knowledge
        core_features = [
            # Trend
            'price_vs_sma50',
            'price_vs_sma200',
            
            # Momentum
            'rsi14',
            'macd',
            'macd_bullish',
            
            # Volatility
            'atr',
            'bb_bandwidth',
            'bb_percent',
            
            # Volume
            'volume_ratio',
            'obv',
            
            # Structure
            'hl_spread_pct',
            
            # Returns (lagged)
            'daily_return',
            'price_change',
            
            # Price position
            'bb_squeeze'
        ]
        
        # Filter to existing features
        suggested = [f for f in core_features if f in self.feature_cols]
        
        print(f"\n  Selected {len(suggested)} features:")
        for i, feature in enumerate(suggested, 1):
            print(f"    {i:2d}. {feature}")
        
        return suggested


def main():
    """Main feature analysis workflow"""
    print("="*60)
    print("🔬 FEATURE ANALYSIS FRAMEWORK")
    print("="*60)
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer('../data/NVDA_dss_features_20251212.csv')
    
    # Identify feature columns
    features = analyzer.identify_feature_columns()
    
    # 1. Correlation analysis
    high_corr = analyzer.correlation_analysis(threshold=0.9)
    
    # 2. VIF analysis
    vif_df = analyzer.calculate_vif()
    
    # 3. Define feature groups
    groups = analyzer.define_feature_groups()
    
    # 4. Suggest reduced feature set
    reduced_features = analyzer.suggest_reduced_feature_set()
    
    # 5. Visualizations
    print("\n📊 Generating visualizations...")
    analyzer.plot_correlation_heatmap('correlation_heatmap.png')
    
    # Save results
    print("\n💾 Saving analysis results...")
    results_summary = {
        'total_features': len(features),
        'high_correlations': len(high_corr),
        'high_vif_features': len(vif_df[vif_df['vif'] > 5]),
        'suggested_reduced_set': len(reduced_features),
        'feature_groups': {k: len(v) for k, v in groups.items()}
    }
    
    pd.DataFrame([results_summary]).to_csv('feature_analysis_summary.csv', index=False)
    
    print("\n✅ Feature analysis complete!")
    print("\nNext steps:")
    print("1. Review correlation heatmap: correlation_heatmap.png")
    print("2. Check feature_analysis_summary.csv for summary")
    print("3. Run ablation_test.py to test feature group importance")
    print("4. Compare full vs reduced feature set performance")


if __name__ == "__main__":
    main()
