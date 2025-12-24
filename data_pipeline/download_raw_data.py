#!/usr/bin/env python3
"""
Script de download raw data tu Yahoo Finance cho cac semiconductor stocks
Download data tu 2010-01-01 den 2025-12-31
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# Danh sach cac semiconductor stocks
STOCKS = ['NVDA', 'AMD', 'MU', 'INTC', 'QCOM', 'TSM']

# Date range: 2010-2025
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2025, 12, 31)

# Output directory - su dung absolute path de dam bao dung
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "data")
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")


def download_stock_data(ticker: str, start_date: datetime, end_date: datetime, output_dir: str) -> bool:
    """
    Download raw data cho mot stock
    
    Args:
        ticker: Stock symbol
        start_date: Ngay bat dau
        end_date: Ngay ket thuc
        output_dir: Thu muc luu file
        
    Returns:
        True neu thanh cong, False neu that bai
    """
    print(f"\nDownloading {ticker}...")
    
    try:
        # Download data tu Yahoo Finance
        # Su dung auto_adjust=False de co ca Close va Adj Close
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)
        
        if df is None or df.empty:
            print(f"  Warning: Khong co data cho {ticker}")
            return False
        
        # Reset index de co Date column
        df = df.reset_index()
        if 'Date' not in df.columns:
            df.index.name = 'Date'
            df = df.reset_index()
        
        # Kiem tra va them Adj Close neu chua co
        # yfinance co the tra ve 'Adj Close' hoac khong, tuy thuoc vao auto_adjust
        if 'Adj Close' not in df.columns:
            # Neu khong co Adj Close, su dung Close lam Adj Close
            # (trong truong hop khong co stock splits/dividends, chung giong nhau)
            df['Adj Close'] = df['Close']
            print(f"  Note: Khong co Adj Close, su dung Close thay the")
        
        # Loai bo cac cot khong can thiet (Dividends, Stock Splits)
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = df[[col for col in columns_to_keep if col in df.columns]]
        
        # Tao thu muc neu chua co (absolute path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Luu file (absolute path)
        date_str = datetime.now().strftime('%Y%m%d')
        filename = os.path.join(output_dir, f"{ticker}_raw_{date_str}.csv")
        df.to_csv(filename, index=False)
        
        # Hien thi absolute path de de kiem tra
        abs_filename = os.path.abspath(filename)
        print(f"  OK: Saved {len(df)} rows to {abs_filename}")
        print(f"      Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        return True
        
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return False


def main():
    """Main function"""
    print("="*60)
    print("Download Raw Stock Data from Yahoo Finance")
    print("="*60)
    print(f"Date Range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Stocks: {', '.join(STOCKS)}")
    abs_raw_dir = os.path.abspath(RAW_DIR)
    print(f"Output Directory: {abs_raw_dir}")
    
    # Tao thu muc output (absolute path)
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Created directory: {abs_raw_dir}")
    
    # Download cho tung stock
    success_count = 0
    for ticker in STOCKS:
        if download_stock_data(ticker, START_DATE, END_DATE, RAW_DIR):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"Download Complete: {success_count}/{len(STOCKS)} stocks")
    print("="*60)
    abs_raw_dir = os.path.abspath(RAW_DIR)
    print(f"\nRaw data saved to: {abs_raw_dir}")
    print("\nNext step: Run add_features.py to add technical indicators")


if __name__ == "__main__":
    main()

