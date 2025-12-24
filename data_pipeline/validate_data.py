#!/usr/bin/env python3
"""
Script de validate data da download co dung voi Yahoo Finance khong
Kiem tra:
- So luong rows
- Date range
- Gia tri mau (Open, High, Low, Close, Volume) tai mot so diem
- So sanh voi data tu Yahoo Finance
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

# Danh sach stocks
STOCKS = ['NVDA', 'AMD', 'MU', 'INTC', 'QCOM', 'TSM']

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw")
FEATURE_DIR = os.path.join(WORKSPACE_ROOT, "data", "Feature")


def validate_raw_data(ticker: str, raw_dir: str) -> dict:
    """
    Validate raw data da download voi Yahoo Finance
    
    Args:
        ticker: Stock symbol
        raw_dir: Thu muc chua raw data
        
    Returns:
        Dictionary chua ket qua validation
    """
    print(f"\n{'='*60}")
    print(f"Validating {ticker} Raw Data")
    print(f"{'='*60}")
    
    result = {
        'ticker': ticker,
        'raw_file_exists': False,
        'yahoo_data_available': False,
        'row_count_match': False,
        'date_range_match': False,
        'sample_values_match': False,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Tim file raw data moi nhat
        raw_files = [f for f in os.listdir(raw_dir) if f.startswith(f"{ticker}_raw_") and f.endswith(".csv")]
        if not raw_files:
            result['errors'].append(f"Khong tim thay raw data file cho {ticker}")
            print(f"  Error: {result['errors'][-1]}")
            return result
        
        result['raw_file_exists'] = True
        latest_file = sorted(raw_files)[-1]
        raw_path = os.path.join(raw_dir, latest_file)
        
        # Load raw data
        df_raw = pd.read_csv(raw_path)
        if 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], utc=True)
        
        print(f"  Raw file: {latest_file}")
        print(f"  Rows: {len(df_raw)}")
        print(f"  Date range: {df_raw['Date'].min().date()} to {df_raw['Date'].max().date()}")
        
        # Download data tu Yahoo Finance de so sanh
        print(f"\n  Downloading data from Yahoo Finance for comparison...")
        stock = yf.Ticker(ticker)
        start_date = df_raw['Date'].min().date()
        end_date = df_raw['Date'].max().date()
        
        df_yahoo = stock.history(start=start_date, end=end_date, auto_adjust=False)
        
        if df_yahoo is None or df_yahoo.empty:
            result['errors'].append(f"Khong the download data tu Yahoo Finance cho {ticker}")
            print(f"  Error: {result['errors'][-1]}")
            return result
        
        result['yahoo_data_available'] = True
        
        # Reset index de co Date column
        df_yahoo = df_yahoo.reset_index()
        if 'Date' not in df_yahoo.columns:
            df_yahoo.index.name = 'Date'
            df_yahoo = df_yahoo.reset_index()
        
        if 'Date' in df_yahoo.columns:
            df_yahoo['Date'] = pd.to_datetime(df_yahoo['Date'], utc=True)
        
        # Them Adj Close neu chua co
        if 'Adj Close' not in df_yahoo.columns:
            df_yahoo['Adj Close'] = df_yahoo['Close']
        
        print(f"  Yahoo Finance rows: {len(df_yahoo)}")
        print(f"  Yahoo Finance date range: {df_yahoo['Date'].min().date()} to {df_yahoo['Date'].max().date()}")
        
        # So sanh so luong rows
        row_diff = abs(len(df_raw) - len(df_yahoo))
        if row_diff == 0:
            result['row_count_match'] = True
            print(f"  Row count: MATCH ({len(df_raw)} rows)")
        else:
            result['warnings'].append(f"Row count khac nhau: Raw={len(df_raw)}, Yahoo={len(df_yahoo)}, Diff={row_diff}")
            print(f"  Row count: DIFFERENT (Raw: {len(df_raw)}, Yahoo: {len(df_yahoo)}, Diff: {row_diff})")
        
        # So sanh date range
        raw_start = df_raw['Date'].min().date()
        raw_end = df_raw['Date'].max().date()
        yahoo_start = df_yahoo['Date'].min().date()
        yahoo_end = df_yahoo['Date'].max().date()
        
        if raw_start == yahoo_start and raw_end == yahoo_end:
            result['date_range_match'] = True
            print(f"  Date range: MATCH ({raw_start} to {raw_end})")
        else:
            result['warnings'].append(f"Date range khac nhau: Raw={raw_start} to {raw_end}, Yahoo={yahoo_start} to {yahoo_end}")
            print(f"  Date range: DIFFERENT")
            print(f"    Raw: {raw_start} to {raw_end}")
            print(f"    Yahoo: {yahoo_start} to {yahoo_end}")
        
        # Merge de so sanh gia tri
        df_merged = pd.merge(
            df_raw[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']],
            df_yahoo[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']],
            on='Date',
            suffixes=('_raw', '_yahoo'),
            how='inner'
        )
        
        print(f"\n  Comparing values on {len(df_merged)} common dates...")
        
        # So sanh gia tri tai mot so diem (dau, giua, cuoi)
        sample_indices = [0, len(df_merged)//2, len(df_merged)-1] if len(df_merged) > 2 else [0]
        sample_dates = [df_merged.iloc[i]['Date'].date() for i in sample_indices]
        
        all_match = True
        columns_to_check = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        for idx, date in zip(sample_indices, sample_dates):
            row = df_merged.iloc[idx]
            print(f"\n    Sample date: {date}")
            
            for col in columns_to_check:
                raw_val = row[f'{col}_raw']
                yahoo_val = row[f'{col}_yahoo']
                
                # So sanh voi tolerance cho float
                if pd.isna(raw_val) and pd.isna(yahoo_val):
                    match = True
                elif pd.isna(raw_val) or pd.isna(yahoo_val):
                    match = False
                elif col == 'Volume':
                    # Volume co the khac nhau do rounding, so sanh voi tolerance 1%
                    match = abs(raw_val - yahoo_val) / max(abs(yahoo_val), 1) < 0.01
                else:
                    # Price values, so sanh voi tolerance 0.01%
                    match = abs(raw_val - yahoo_val) / max(abs(yahoo_val), 0.0001) < 0.0001
                
                if not match:
                    all_match = False
                    diff_pct = abs(raw_val - yahoo_val) / max(abs(yahoo_val), 0.0001) * 100
                    print(f"      {col}: MISMATCH - Raw={raw_val:.6f}, Yahoo={yahoo_val:.6f}, Diff={diff_pct:.4f}%")
                else:
                    print(f"      {col}: OK")
        
        # So sanh tong the
        print(f"\n  Overall comparison:")
        total_mismatches = 0
        for col in columns_to_check:
            raw_col = f'{col}_raw'
            yahoo_col = f'{col}_yahoo'
            
            if col == 'Volume':
                # Volume: tolerance 1%
                diff_pct = ((df_merged[raw_col] - df_merged[yahoo_col]).abs() / df_merged[yahoo_col].abs().replace(0, 1) * 100)
                mismatches = (diff_pct > 1.0).sum()
            else:
                # Price: tolerance 0.01%
                diff_pct = ((df_merged[raw_col] - df_merged[yahoo_col]).abs() / df_merged[yahoo_col].abs().replace(0, 0.0001) * 100)
                mismatches = (diff_pct > 0.01).sum()
            
            total_mismatches += mismatches
            if mismatches > 0:
                print(f"    {col}: {mismatches}/{len(df_merged)} mismatches ({mismatches/len(df_merged)*100:.2f}%)")
                result['warnings'].append(f"{col}: {mismatches} mismatches")
            else:
                print(f"    {col}: All values match")
        
        if total_mismatches == 0:
            result['sample_values_match'] = True
            print(f"\n  Result: ALL VALUES MATCH")
        else:
            print(f"\n  Result: {total_mismatches} TOTAL MISMATCHES FOUND")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Error validating {ticker}: {str(e)}")
        print(f"  Error: {result['errors'][-1]}")
        import traceback
        traceback.print_exc()
        return result


def validate_feature_data(ticker: str, feature_dir: str) -> dict:
    """
    Validate feature data co dung format khong
    
    Args:
        ticker: Stock symbol
        feature_dir: Thu muc chua feature data
        
    Returns:
        Dictionary chua ket qua validation
    """
    print(f"\n{'='*60}")
    print(f"Validating {ticker} Feature Data")
    print(f"{'='*60}")
    
    result = {
        'ticker': ticker,
        'feature_file_exists': False,
        'has_required_columns': False,
        'has_valid_data': False,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Tim file feature moi nhat
        feature_files = [f for f in os.listdir(feature_dir) if f.startswith(f"{ticker}_dss_features_") and f.endswith(".csv")]
        if not feature_files:
            result['errors'].append(f"Khong tim thay feature file cho {ticker}")
            print(f"  Error: {result['errors'][-1]}")
            return result
        
        result['feature_file_exists'] = True
        latest_file = sorted(feature_files)[-1]
        feature_path = os.path.join(feature_dir, latest_file)
        
        # Load feature data
        df = pd.read_csv(feature_path, index_col=0)
        
        print(f"  Feature file: {latest_file}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Kiem tra cac cot can thiet (tham khao NVDA_dss_features_20251212.csv)
        required_columns = [
            'Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume',
            'daily_return', 'sma50', 'sma200', 'rsi14', 'macd', 'macd_signal', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'bb_percent', 'atr', 'obv',
            'price_change', 'hl_spread', 'hl_spread_pct', 'oc_spread', 'oc_spread_pct',
            'volume_sma20', 'volume_ratio', 'price_vs_sma50', 'price_vs_sma200',
            'rsi_overbought', 'rsi_oversold', 'macd_bullish', 'bb_squeeze'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            result['errors'].append(f"Thieu cac cot: {missing_columns}")
            print(f"  Error: Thieu {len(missing_columns)} cot: {missing_columns[:5]}...")
        else:
            result['has_required_columns'] = True
            print(f"  Required columns: All present ({len(required_columns)} columns)")
        
        # Kiem tra data co hop le khong (khong co NaN o cac cot quan trong)
        important_columns = ['Date', 'Adj Close', 'Close', 'daily_return', 'rsi14', 'macd']
        nan_counts = {}
        for col in important_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    nan_counts[col] = nan_count
        
        if nan_counts:
            result['warnings'].append(f"Co NaN trong cac cot: {nan_counts}")
            print(f"  Warning: Co NaN trong {len(nan_counts)} cot quan trong")
            for col, count in nan_counts.items():
                print(f"    {col}: {count} NaN values")
        else:
            result['has_valid_data'] = True
            print(f"  Data validity: No NaN in important columns")
        
        # Kiem tra date range
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Error validating feature data for {ticker}: {str(e)}")
        print(f"  Error: {result['errors'][-1]}")
        import traceback
        traceback.print_exc()
        return result


def main():
    """Main function"""
    print("="*60)
    print("Validate Downloaded Data vs Yahoo Finance")
    print("="*60)
    print(f"Stocks: {', '.join(STOCKS)}")
    abs_raw_dir = os.path.abspath(RAW_DIR)
    abs_feature_dir = os.path.abspath(FEATURE_DIR)
    print(f"Raw Data Directory: {abs_raw_dir}")
    print(f"Feature Data Directory: {abs_feature_dir}")
    
    # Kiem tra thu muc
    if not os.path.exists(RAW_DIR):
        print(f"\nError: Thu muc {abs_raw_dir} khong ton tai")
        print("Chay download_raw_data.py truoc!")
        return
    
    # Validate raw data
    print("\n" + "="*60)
    print("VALIDATING RAW DATA")
    print("="*60)
    
    raw_results = []
    for ticker in STOCKS:
        result = validate_raw_data(ticker, RAW_DIR)
        raw_results.append(result)
    
    # Validate feature data (neu co)
    if os.path.exists(FEATURE_DIR):
        print("\n" + "="*60)
        print("VALIDATING FEATURE DATA")
        print("="*60)
        
        feature_results = []
        for ticker in STOCKS:
            result = validate_feature_data(ticker, FEATURE_DIR)
            feature_results.append(result)
    else:
        print(f"\nNote: Thu muc {abs_feature_dir} khong ton tai")
        print("Chay add_features.py de tao feature data")
        feature_results = []
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print("\nRaw Data Validation:")
    for result in raw_results:
        status = "PASS" if (result['row_count_match'] and result['date_range_match'] and result['sample_values_match']) else "FAIL"
        print(f"  {result['ticker']}: {status}")
        if result['errors']:
            for err in result['errors']:
                print(f"    Error: {err}")
        if result['warnings']:
            for warn in result['warnings'][:3]:  # Chi hien thi 3 warnings dau tien
                print(f"    Warning: {warn}")
    
    if feature_results:
        print("\nFeature Data Validation:")
        for result in feature_results:
            status = "PASS" if (result['has_required_columns'] and result['has_valid_data']) else "FAIL"
            print(f"  {result['ticker']}: {status}")
            if result['errors']:
                for err in result['errors']:
                    print(f"    Error: {err}")
            if result['warnings']:
                for warn in result['warnings'][:3]:
                    print(f"    Warning: {warn}")
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

