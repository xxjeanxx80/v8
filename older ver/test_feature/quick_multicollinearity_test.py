#!/usr/bin/env python3
"""
Quick Multicollinearity Test
Phan tich VIF va correlation nhanh de xac dinh features redundant
Khong can train model, chi can load data va tinh toan
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v4_multistock'))
sys.path.append('..')

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings("ignore")

# Load feature set tu v5
v5_path = os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v5_multistock', 'nvda_lstm_v5_multistock.py')
sys.path.insert(0, os.path.dirname(v5_path))
try:
    from nvda_lstm_v5_multistock import load_optimized_features
except:
    # Fallback: doc tu config
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

import config

class QuickMulticollinearityTester:
    def __init__(self, data_dir=None):
        """Khoi tao voi data directory"""
        self.data_dir = data_dir or config.DATA_DIR
        self.v5_features = load_optimized_features()
        self.df = None
        self.feature_cols = []
        self.vif_results = None
        self.correlation_results = None
        
    def load_data(self):
        """Load data va filter theo v5 features"""
        print("Dang load data...")
        
        # Load data tu NVDA file
        nvda_file = os.path.join(self.data_dir, 'NVDA_dss_features_20251212.csv')
        if not os.path.exists(nvda_file):
            raise FileNotFoundError(f"Khong tim thay file: {nvda_file}")
        
        self.df = pd.read_csv(nvda_file)
        
        # Filter features theo v5 feature set
        available_features = [f for f in self.v5_features if f in self.df.columns]
        missing_features = set(self.v5_features) - set(available_features)
        
        if missing_features:
            print(f"Canh bao: Thieu features: {missing_features}")
        
        self.feature_cols = available_features
        print(f"Su dung {len(self.feature_cols)} features tu v5 feature set")
        
        return self.df, self.feature_cols
    
    def calculate_vif(self):
        """Tinh VIF cho tat ca features"""
        print("\nDang tinh VIF...")
        
        # Chuan bi data
        X = self.df[self.feature_cols].dropna()
        
        if len(X) == 0:
            raise ValueError("Khong co du lieu sau khi dropna")
        
        # Loai bo features co variance = 0
        X = X.loc[:, X.var() > 1e-6]
        self.feature_cols = list(X.columns)
        
        X_const = add_constant(X)
        
        # Tinh VIF cho tung feature
        vif_data = []
        for i, feature in enumerate(X.columns):
            try:
                vif = variance_inflation_factor(X_const.values, i+1)
                vif_data.append({
                    'feature': feature,
                    'vif': vif if not np.isinf(vif) and not np.isnan(vif) else 999.0
                })
            except Exception as e:
                print(f"  Loi khi tinh VIF cho {feature}: {e}")
                vif_data.append({
                    'feature': feature,
                    'vif': 999.0
                })
        
        self.vif_results = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
        
        # Hien thi ket qua
        print(f"\nVIF Analysis ({len(self.vif_results)} features):")
        high_vif = self.vif_results[self.vif_results['vif'] > 5]
        if len(high_vif) > 0:
            print(f"  Co {len(high_vif)} features co VIF > 5:")
            for _, row in high_vif.iterrows():
                print(f"    - {row['feature']}: VIF = {row['vif']:.2f}")
        else:
            print("  Khong co features nao co VIF > 5")
        
        avg_vif = self.vif_results['vif'].mean()
        print(f"  Average VIF: {avg_vif:.2f}")
        print(f"  Max VIF: {self.vif_results['vif'].max():.2f}")
        
        return self.vif_results
    
    def calculate_correlation(self, threshold=0.85):
        """Tinh correlation matrix va tim cac cap redundant"""
        print(f"\nDang tinh correlation (threshold = {threshold})...")
        
        # Tinh correlation matrix
        corr_matrix = self.df[self.feature_cols].corr()
        
        # Tim cac cap correlation cao
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        self.correlation_results = {
            'matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'threshold': threshold
        }
        
        # Hien thi ket qua
        if high_corr_pairs:
            print(f"  Tim thay {len(high_corr_pairs)} cap features co correlation > {threshold}:")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]:
                print(f"    - {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print(f"  Khong co cap nao co correlation > {threshold}")
        
        return self.correlation_results
    
    def identify_redundant_groups(self):
        """Xac dinh cac nhom features redundant"""
        print("\nDang xac dinh nhom features redundant...")
        
        if self.correlation_results is None:
            self.calculate_correlation()
        
        # Tao graph de tim nhom features lien ket
        from collections import defaultdict
        
        groups = defaultdict(list)
        processed = set()
        
        # Nhom theo correlation
        for pair in self.correlation_results['high_corr_pairs']:
            f1, f2 = pair['feature1'], pair['feature2']
            # Tim nhom da co hoac tao moi
            found_group = None
            for group_name, features in groups.items():
                if f1 in features or f2 in features:
                    found_group = group_name
                    break
            
            if found_group:
                if f1 not in groups[found_group]:
                    groups[found_group].append(f1)
                if f2 not in groups[found_group]:
                    groups[found_group].append(f2)
            else:
                new_group = f"group_{len(groups)+1}"
                groups[new_group] = [f1, f2]
        
        # Nhom theo ten (MACD, return, spread, etc.)
        name_groups = {
            'macd': ['macd', 'macd_signal', 'macd_hist', 'macd_bullish'],
            'return': ['daily_return', 'price_change', 'return_3d', 'return_5d', 'return_10d', 'return_20d'],
            'spread': ['hl_spread_pct', 'oc_spread', 'oc_spread_pct'],
            'bb': ['bb_bandwidth', 'bb_percent', 'bb_squeeze'],
            'rsi': ['rsi14', 'rsi_overbought', 'rsi_oversold'],
            'volume': ['volume_ratio', 'obv', 'volume_sma20'],
            'sector': ['sox_beta', 'sox_correlation']
        }
        
        # Filter chi lay features co trong dataset
        filtered_name_groups = {}
        for group_name, features in name_groups.items():
            existing = [f for f in features if f in self.feature_cols]
            if len(existing) > 1:
                filtered_name_groups[group_name] = existing
        
        print(f"\nNhom features theo ten:")
        for group_name, features in filtered_name_groups.items():
            print(f"  {group_name}: {features}")
        
        return filtered_name_groups, dict(groups)
    
    def suggest_features_to_remove(self):
        """De xuat features nen loai bo"""
        print("\nDang de xuat features nen loai bo...")
        
        if self.vif_results is None:
            self.calculate_vif()
        
        if self.correlation_results is None:
            self.calculate_correlation()
        
        # Lay features co VIF cao
        high_vif_features = self.vif_results[self.vif_results['vif'] > 5]['feature'].tolist()
        
        # Lay features trong cac cap correlation cao
        high_corr_features = set()
        for pair in self.correlation_results['high_corr_pairs']:
            high_corr_features.add(pair['feature1'])
            high_corr_features.add(pair['feature2'])
        
        # De xuat: features co VIF cao HOAC trong nhom correlation cao
        candidates_to_remove = list(set(high_vif_features) | high_corr_features)
        
        # Sap xep theo uu tien: VIF cao nhat truoc
        vif_dict = dict(zip(self.vif_results['feature'], self.vif_results['vif']))
        candidates_to_remove.sort(key=lambda x: vif_dict.get(x, 0), reverse=True)
        
        suggestions = []
        for feature in candidates_to_remove:
            vif = vif_dict.get(feature, 0)
            in_corr = feature in high_corr_features
            suggestions.append({
                'feature': feature,
                'vif': vif,
                'in_high_correlation': in_corr,
                'priority': 'high' if vif > 10 or in_corr else 'medium'
            })
        
        suggestions_df = pd.DataFrame(suggestions)
        
        print(f"\nDe xuat loai bo {len(suggestions_df)} features:")
        high_priority = suggestions_df[suggestions_df['priority'] == 'high']
        if len(high_priority) > 0:
            print(f"  Uu tien cao ({len(high_priority)} features):")
            for _, row in high_priority.iterrows():
                print(f"    - {row['feature']}: VIF={row['vif']:.2f}, HighCorr={row['in_high_correlation']}")
        
        medium_priority = suggestions_df[suggestions_df['priority'] == 'medium']
        if len(medium_priority) > 0:
            print(f"  Uu tien trung binh ({len(medium_priority)} features):")
            for _, row in medium_priority.head(5).iterrows():
                print(f"    - {row['feature']}: VIF={row['vif']:.2f}")
        
        return suggestions_df
    
    def run_analysis(self):
        """Chay tat ca phan tich"""
        print("="*60)
        print("QUICK MULTICOLLINEARITY ANALYSIS")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Tinh VIF
        vif_df = self.calculate_vif()
        
        # Tinh correlation
        corr_results = self.calculate_correlation(threshold=0.85)
        
        # Xac dinh nhom redundant
        name_groups, corr_groups = self.identify_redundant_groups()
        
        # De xuat features nen loai bo
        suggestions = self.suggest_features_to_remove()
        
        # Luu ket qua
        self.save_results(vif_df, corr_results, suggestions)
        
        return vif_df, corr_results, suggestions
    
    def save_results(self, vif_df, corr_results, suggestions):
        """Luu tat ca ket qua"""
        results_dir = os.path.join(os.path.dirname(__file__), config.RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        
        # Luu VIF analysis
        vif_path = os.path.join(results_dir, 'quick_vif_analysis.csv')
        vif_df.to_csv(vif_path, index=False)
        print(f"\nDa luu VIF analysis vao {vif_path}")
        
        # Luu correlation pairs
        if corr_results['high_corr_pairs']:
            corr_path = os.path.join(results_dir, 'quick_correlation_pairs.csv')
            corr_df = pd.DataFrame(corr_results['high_corr_pairs'])
            corr_df.to_csv(corr_path, index=False)
            print(f"Da luu correlation pairs vao {corr_path}")
        
        # Luu features to remove
        remove_path = os.path.join(results_dir, 'features_to_remove.csv')
        suggestions.to_csv(remove_path, index=False)
        print(f"Da luu features to remove vao {remove_path}")
        
        # Luu correlation matrix
        corr_matrix_path = os.path.join(results_dir, 'quick_correlation_matrix.csv')
        corr_results['matrix'].to_csv(corr_matrix_path)
        print(f"Da luu correlation matrix vao {corr_matrix_path}")


def main():
    """Main workflow"""
    tester = QuickMulticollinearityTester()
    vif_df, corr_results, suggestions = tester.run_analysis()
    
    print("\n" + "="*60)
    print("HOAN THANH QUICK MULTICOLLINEARITY ANALYSIS")
    print("="*60)
    print(f"\nTong ket:")
    print(f"  - So features: {len(tester.feature_cols)}")
    print(f"  - Features co VIF > 5: {len(vif_df[vif_df['vif'] > 5])}")
    print(f"  - Cap correlation cao: {len(corr_results['high_corr_pairs'])}")
    print(f"  - Features nen xem xet loai bo: {len(suggestions)}")


if __name__ == "__main__":
    main()

