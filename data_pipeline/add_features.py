#!/usr/bin/env python3
"""
Script de them cac technical indicators vao raw data
Tinh toan features truc tiep trong script nay
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


# ==================== Feature Calculation Functions ====================
def add_returns(df):
    """Them daily return"""
    df["daily_return"] = df["Adj Close"].pct_change()
    return df


def add_trend_indicators(df):
    """
    Them trend indicators (SMA relative features)
    V7.2: Chi tinh relative features, khong luu absolute SMA values
    """
    # Tinh SMA de su dung cho relative features
    sma50 = df["Adj Close"].rolling(window=50).mean()
    sma200 = df["Adj Close"].rolling(window=200).mean()
    
    # Chi luu relative features
    df["price_vs_sma50"] = (df["Adj Close"] - sma50) / sma50.replace(0, pd.NA)
    df["price_vs_sma200"] = (df["Adj Close"] - sma200) / sma200.replace(0, pd.NA)
    
    # Khong luu sma50, sma200 (absolute values)
    return df


def compute_rsi(series, period=14):
    """Tinh RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_momentum_indicators(df):
    """
    Them momentum indicators (RSI, MACD)
    V7.2: MACD duoc chuẩn hóa theo giá để tránh absolute values
    """
    df["rsi14"] = compute_rsi(df["Adj Close"], 14)
    ema12 = df["Adj Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Adj Close"].ewm(span=26, adjust=False).mean()
    
    # V7.2: Chuan hoa MACD theo gia de tranh absolute values
    # Tinh MACD ratio (relative to price) thay vi absolute difference
    macd_abs = ema12 - ema26
    df["macd"] = macd_abs / df["Adj Close"].replace(0, pd.NA)
    
    # Tinh signal tu MACD ratio (relative)
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_volatility_indicators(df, period=20):
    """
    Them volatility indicators (Bollinger Bands, ATR)
    V7.2: Chi tinh relative features, khong luu absolute BB values
    """
    # Tinh BB de su dung cho relative features
    bb_middle = df["Adj Close"].rolling(period).mean()
    bb_std = df["Adj Close"].rolling(period).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Chi luu relative features
    df["bb_bandwidth"] = (bb_upper - bb_lower) / bb_middle.replace(0, pd.NA)
    df["bb_percent"] = (df["Adj Close"] - bb_lower) / (bb_upper - bb_lower).replace(0, pd.NA)
    
    # Khong luu bb_upper, bb_lower, bb_middle (absolute values)
    
    # ATR (Average True Range)
    # V7.2: Chia cho gia de thanh ATR % (relative) thay vi absolute dollar value
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_abs = true_range.rolling(window=14).mean()
    df["atr"] = atr_abs / df["Adj Close"].replace(0, pd.NA)  # ATR as percentage
    return df


def add_obv_relative(df):
    """
    Them OBV relative feature (V7.2 compatible)
    Thay vi OBV cộng dồn tuyệt đối, tinh OBV relative (rolling window)
    """
    vol = df["Volume"].fillna(0).astype("float64")
    change = df["Adj Close"].diff()
    
    # Tính dòng tiền ra/vào mỗi ngày (positive khi giá tăng, negative khi giảm)
    flow = np.where(change > 0, vol, np.where(change < 0, -vol, 0))
    
    # Thay vì cộng dồn mãi mãi, ta tính tổng dòng tiền trong 20 ngày gần nhất
    obv_20d = pd.Series(flow).rolling(window=20).sum()
    
    # Chuẩn hóa theo Volume trung bình để thành số tương đối
    volume_sum_20d = df["Volume"].rolling(window=20).sum()
    df["obv_feature"] = obv_20d / volume_sum_20d.replace(0, pd.NA)
    
    return df


def add_additional_features(df):
    """
    Them cac features bo sung
    V7.2: Chi tinh relative features, khong luu absolute values
    """
    # V7.2: Bo price_change vi trung lap voi daily_return (da co trong add_returns)
    # df["price_change"] = df["Adj Close"].diff()  # REMOVED: Absolute value, trung voi daily_return
    
    # V7.2: Chi giu lai _pct versions, xoa absolute spreads
    # hl_spread va oc_spread la absolute dollar values, chi giu _pct
    df["hl_spread_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["oc_spread_pct"] = (df["Close"] - df["Open"]) / df["Open"]
    # Khong luu hl_spread, oc_spread (absolute values)
    
    # Tinh volume_ratio, sau do se xoa Volume goc va volume_sma20
    volume_sma20 = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / volume_sma20.replace(0, pd.NA)
    # Khong luu volume_sma20 (absolute value)
    
    # price_vs_sma50 va price_vs_sma200 da duoc tinh trong add_trend_indicators
    # Khong tinh lai o day nua
    
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
        df = add_trend_indicators(df)  # Chi tinh price_vs_sma50, price_vs_sma200
        df = add_momentum_indicators(df)
        df = add_volatility_indicators(df)  # Chi tinh bb_bandwidth, bb_percent
        df = add_obv_relative(df)  # V7.2: OBV relative thay vi absolute
        df = add_additional_features(df)  # Chi tinh volume_ratio
        
        # V7.2: Loai bo triet de cac cot Absolute
        # Xoa tat ca absolute values de Scaler hoat dong hoan hao nhat
        absolute_features_to_drop = [
            # Volume (absolute)
            'Volume',  # Absolute volume, da co volume_ratio
            'volume_sma20',  # Absolute volume SMA, da co volume_ratio
            
            # SMA (absolute prices)
            'sma50',  # Absolute SMA, da co price_vs_sma50
            'sma200',  # Absolute SMA, da co price_vs_sma200
            
            # Bollinger Bands (absolute prices)
            'bb_upper',  # Absolute BB upper, da co bb_percent
            'bb_lower',  # Absolute BB lower, da co bb_percent
            'bb_middle',  # Absolute BB middle, da co bb_percent va bb_bandwidth
            
            # OBV (absolute cumulative)
            'obv',  # Absolute OBV (neu co), da co obv_feature
            
            # Spreads & Changes (absolute dollar values) - them vao de xoa neu lo tao ra
            'hl_spread',  # Absolute High-Low spread, da co hl_spread_pct
            'oc_spread',  # Absolute Open-Close spread, da co oc_spread_pct
            'price_change',  # Absolute price change, da co daily_return
            
            # OHLC prices (absolute) - xoa luon OHLC goc, chi giu Adj Close
            'Open',  # Absolute Open price
            'High',  # Absolute High price
            'Low',  # Absolute Low price
            'Close'  # Absolute Close price (chi giu Adj Close)
        ]
        
        # Chi xoa cac features ton tai trong dataframe
        features_to_drop = [f for f in absolute_features_to_drop if f in df.columns]
        if features_to_drop:
            df = df.drop(columns=features_to_drop)
            print(f"  Removed absolute features ({len(features_to_drop)}): {', '.join(features_to_drop)}")
        
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

