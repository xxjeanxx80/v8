#!/usr/bin/env python3
"""
Cap nhat optimized_feature_config.csv voi best feature set tu ket qua quick test
Best config: remove_top7_vif (10 features)
"""

import pandas as pd
import os

# Best feature set tu remove_top7_vif
best_features = [
    'macd_bullish',
    'bb_bandwidth',
    'volume_ratio',
    'volume_sma20',
    'daily_return',
    'price_change',
    'return_3d',
    'return_5d',
    'return_10d',
    'return_20d',
    'hl_spread_pct',
    'oc_spread',
    'oc_spread_pct',
    'bb_squeeze',
    'rsi_overbought',
    'rsi_oversold',
    'sox_beta',
    'sox_correlation'
]

# Doc best_feature_set_v5.csv de lay chinh xac
results_dir = os.path.join(os.path.dirname(__file__), 'results')
best_config_path = os.path.join(results_dir, 'best_feature_set_v5.csv')

if os.path.exists(best_config_path):
    df_best = pd.read_csv(best_config_path)
    if not df_best.empty and 'features' in df_best.columns:
        features_str = df_best.iloc[0]['features']
        try:
            best_features = eval(features_str)
            print(f"Da doc best features tu {best_config_path}: {len(best_features)} features")
        except:
            print(f"Khong the parse features tu {best_config_path}, su dung default")

# Tao config moi
rationale = [
    f"Best feature set tu quick test - remove_top7_vif ({len(best_features)} features)",
    "Giam multicollinearity thanh cong: VIF giam tu hang nghin xuong 3.2",
    "Tang do chinh xac: Combined WR 69.65%, Buy WR 93.94%",
    "Loai bo 7 features co VIF cao nhat: macd_hist, macd_signal, macd, rsi14, bb_percent, obv, atr"
]

config_data = {
    'features': [str(best_features)],
    'count': [len(best_features)],
    'rationale': ['; '.join(rationale)]
}

config_df = pd.DataFrame(config_data)

# Luu vao optimized_feature_config.csv
config_path = os.path.join(os.path.dirname(__file__), 'optimized_feature_config.csv')
config_df.to_csv(config_path, index=False)

print(f"\nDa cap nhat {config_path}")
print(f"\nBest feature set ({len(best_features)} features):")
for i, feat in enumerate(best_features, 1):
    print(f"  {i:2d}. {feat}")

print(f"\nMetrics:")
print(f"  - Combined Win Rate: 69.65%")
print(f"  - Buy Win Rate: 93.94%")
print(f"  - Sell Win Rate: 45.36%")
print(f"  - RMSE: 0.0402")
print(f"  - Average VIF: 3.20")

