#!/usr/bin/env python3
"""
Script to download historical data for semiconductor stocks
Download data from 2010-01-01 to 2025-12-31
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# Semiconductor stocks to download
stocks = ['NVDA', 'AMD', 'MU', 'INTC', 'QCOM', 'TSM']

# Download parameters: 2010-2025
start_date = datetime(2010, 1, 1)
end_date = datetime(2025, 12, 31)

print("Downloading semiconductor stock data from 2010-01-01 to 2025-12-31...")

for ticker in stocks:
    print(f"\nDownloading {ticker}...")
    
    try:
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        # Save CSV to data folder
        data_dir = "../data" if os.path.exists("../data") else "data"
        os.makedirs(data_dir, exist_ok=True)
        filename = f"{data_dir}/{ticker}_raw_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename)
        
        print(f"✅ Saved {len(df)} rows to {filename}")
        print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")

print("\n✅ Download complete!")
print("\nNext steps:")
print("1. Run the feature generation script on each CSV file")
print("2. Rename files to match pattern: {TICKER}_dss_features_YYYYMMDD.csv")
print("3. Run nvda_lstm_multistock.py")
