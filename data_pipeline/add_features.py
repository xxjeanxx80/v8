#!/usr/bin/env python3
"""
Script de them cac technical indicators vao raw data
Su dung logic tu nvda_dss_integrated.py de tinh toan features
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Them parent directory vao path de import nvda_dss_integrated
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nvda_dss_integrated import (
        add_returns,
        add_trend_indicators,
        add_momentum_indicators,
        add_volatility_indicators,
        add_obv,
        add_additional_features
    )
except ImportError:
    print("Error: Khong tim thay nvda_dss_integrated.py")
    print("Dang su dung fallback functions...")
    
    # Fallback functions neu khong import duoc
    def add_returns(df):
        df["daily_return"] = df["Adj Close"].pct_change()
        return df
    
    def add_trend_indicators(df):
        df["sma50"] = df["Adj Close"].rolling(window=50).mean()
        df["sma200"] = df["Adj Close"].rolling(window=200).mean()
        return df
    
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def add_momentum_indicators(df):
        df["rsi14"] = compute_rsi(df["Adj Close"], 14)
        ema12 = df["Adj Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Adj Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df
    
    def add_volatility_indicators(df, period=20):
        df["bb_middle"] = df["Adj Close"].rolling(period).mean()
        bb_std = df["Adj Close"].rolling(period).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, pd.NA)
        df["bb_percent"] = (df["Adj Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, pd.NA)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()
        return df
    
    def add_obv(df):
        vol = df["Volume"].fillna(0).astype("int64")
        close = df["Close"].ffill()
        obv = [0]
        for i in range(1, len(df)):
            if close.iat[i] > close.iat[i - 1]:
                obv.append(obv[-1] + int(vol.iat[i]))
            elif close.iat[i] < close.iat[i - 1]:
                obv.append(obv[-1] - int(vol.iat[i]))
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        return df
    
    def add_additional_features(df):
        df["price_change"] = df["Adj Close"].diff()
        df["hl_spread"] = df["High"] - df["Low"]
        df["hl_spread_pct"] = (df["High"] - df["Low"]) / df["Close"]
        df["oc_spread"] = df["Close"] - df["Open"]
        df["oc_spread_pct"] = (df["Close"] - df["Open"]) / df["Open"]
        df["volume_sma20"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma20"]
        df["price_vs_sma50"] = (df["Adj Close"] - df["sma50"]) / df["sma50"]
        df["price_vs_sma200"] = (df["Adj Close"] - df["sma200"]) / df["sma200"]
        df["rsi_overbought"] = (df["rsi14"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi14"] < 30).astype(int)
        df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)
        df["bb_squeeze"] = (df["bb_bandwidth"] < df["bb_bandwidth"].rolling(50).mean() * 0.8).astype(int)
        return df


# Danh sach stocks
STOCKS = ['NVDA', 'AMD', 'MU', 'INTC', 'QCOM', 'TSM']

# Directories - su dung absolute path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "data", "Feature")


def add_features_to_stock(ticker: str, raw_dir: str, output_dir: str) -> bool:
    """
    Them features vao raw data cua mot stock
    
    Args:
        ticker: Stock symbol
        raw_dir: Thu muc chua raw data
        output_dir: Thu muc luu file output
        
    Returns:
        True neu thanh cong, False neu that bai
    """
    print(f"\nProcessing {ticker}...")
    
    try:
        # Tim file raw data moi nhat
        raw_files = [f for f in os.listdir(raw_dir) if f.startswith(f"{ticker}_raw_") and f.endswith(".csv")]
        if not raw_files:
            print(f"  Warning: Khong tim thay raw data cho {ticker}")
            return False
        
        # Lay file moi nhat
        latest_file = sorted(raw_files)[-1]
        raw_path = os.path.join(raw_dir, latest_file)
        
        # Load raw data
        df = pd.read_csv(raw_path)
        
        # Convert Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
        elif df.index.name == 'Date':
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Kiem tra cac cot can thiet
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"  Error: Thieu cac cot: {missing_columns}")
            return False
        
        # Them Adj Close neu chua co (su dung Close)
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
            print(f"  Note: Khong co Adj Close, su dung Close thay the")
        
        # Loai bo cac cot khong can thiet
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        # Sort theo date
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"  Loaded {len(df)} rows from {raw_path}")
        print(f"  Columns: {', '.join(df.columns)}")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Them cac features
        print(f"  Calculating indicators...")
        df = add_returns(df)
        df = add_trend_indicators(df)
        df = add_momentum_indicators(df)
        df = add_volatility_indicators(df)
        df = add_obv(df)
        df = add_additional_features(df)
        
        # Loai bo rows co NaN (do rolling windows)
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with NaN values")
        
        # Luu file
        date_str = datetime.now().strftime('%Y%m%d')
        output_filename = os.path.join(output_dir, f"{ticker}_dss_features_{date_str}.csv")
        df.to_csv(output_filename, index=True)
        
        print(f"  OK: Saved {len(df)} rows with {len(df.columns)} features to {output_filename}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("="*60)
    print("Add Technical Indicators to Raw Stock Data")
    print("="*60)
    print(f"Stocks: {', '.join(STOCKS)}")
    abs_raw_dir = os.path.abspath(RAW_DIR)
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"Raw Data Directory: {abs_raw_dir}")
    print(f"Output Directory: {abs_output_dir}")
    
    # Kiem tra thu muc raw data
    if not os.path.exists(RAW_DIR):
        print(f"\nError: Thu muc {RAW_DIR} khong ton tai")
        print("Chay download_raw_data.py truoc!")
        return
    
    # Tao thu muc output neu chua co
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Xu ly tung stock
    success_count = 0
    for ticker in STOCKS:
        if add_features_to_stock(ticker, RAW_DIR, OUTPUT_DIR):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"Processing Complete: {success_count}/{len(STOCKS)} stocks")
    print("="*60)
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"\nFeature files saved to: {abs_output_dir}")


if __name__ == "__main__":
    main()

