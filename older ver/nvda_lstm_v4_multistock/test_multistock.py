#!/usr/bin/env python3
"""
Test script to create synthetic multi-stock data from NVDA
For testing nvda_lstm_multistock_complete.py
"""

import pandas as pd
import numpy as np
import os
import shutil

print("📊 Creating synthetic multi-stock data for testing...")

# Load NVDA data
data_path = '../data/NVDA_dss_features_20251212.csv' if os.path.exists('../data/NVDA_dss_features_20251212.csv') else 'data/NVDA_dss_features_20251212.csv'
df_nvda = pd.read_csv(data_path)

# Create synthetic variations for other stocks
stocks = ['AMD', 'MU', 'INTC', 'QCOM']

for stock in stocks:
    print(f"\nCreating {stock} data...")
    
    # Copy NVDA structure
    df_stock = df_nvda.copy()
    
    # Add realistic variations
    np.random.seed(hash(stock) % 2**32)  # Reproducible per stock
    
    # Vary price by stock-specific factor
    if stock == 'AMD':
        price_factor = 0.15  # AMD is cheaper
        vol_factor = 1.2     # More volatile
    elif stock == 'MU':
        price_factor = 0.08  
        vol_factor = 1.1
    elif stock == 'INTC':
        price_factor = 0.35
        vol_factor = 0.8     # Less volatile
    else:  # QCOM
        price_factor = 0.18
        vol_factor = 0.9
    
    # Apply variations to price-related columns
    price_cols = ['Adj Close', 'Close', 'Open', 'High', 'Low']
    for col in price_cols:
        if col in df_stock.columns:
            df_stock[col] = df_stock[col] * price_factor * (1 + np.random.normal(0, 0.1, len(df_stock)))
    
    # Vary volume
    if 'Volume' in df_stock.columns:
        df_stock['Volume'] = df_stock['Volume'] * np.random.uniform(0.5, 1.5, len(df_stock))
    
    # Vary returns slightly
    return_cols = ['daily_return', 'future_return']
    for col in return_cols:
        if col in df_stock.columns:
            df_stock[col] = df_stock[col] * vol_factor * (1 + np.random.normal(0, 0.2, len(df_stock)))
    
    # Vary technical indicators
    tech_cols = ['rsi14', 'macd', 'bb_upper', 'bb_lower', 'sma20', 'sma50', 'sma200']
    for col in tech_cols:
        if col in df_stock.columns:
            df_stock[col] = df_stock[col] * (1 + np.random.normal(0, 0.15, len(df_stock)))
    
    # Save as new stock file in data folder
    data_dir = "../data" if os.path.exists("../data") else "data"
    filename = f"{data_dir}/{stock}_dss_features_20251212.csv"
    df_stock.to_csv(filename, index=False)
    print(f"  ✅ Saved {len(df_stock)} rows to {filename}")

print("\n✅ Synthetic multi-stock data created!")
print("\nFiles created:")
for stock in ['NVDA'] + stocks:
    print(f"  - {data_dir}/{stock}_dss_features_20251212.csv")

print("\nNow you can test:")
print("  python nvda_lstm_multistock_complete.py")
