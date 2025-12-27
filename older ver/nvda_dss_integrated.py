#!/usr/bin/env python3
"""
NVDA Stock Price Prediction DSS - Integrated System
Combines all technical indicators and data fetching into a single comprehensive module.

Features:
- Fetch data from Yahoo Finance
- Calculate returns, trend, momentum, volatility, and volume indicators
- Generate comprehensive CSV for DSS analysis
"""

import argparse
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
from dateutil.relativedelta import relativedelta
import yfinance as yf


# ==================== DATA FETCHING ====================

def fetch_prices(ticker: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], datetime, datetime]:
    """Fetch historical prices for `ticker` between start and end.
    
    If `start_date` is None, defaults to 2010-01-01.
    If `end_date` is None, defaults to 2025-12-31.
    Dates must be ISO format YYYY-MM-DD when provided.
    
    Returns (df, start_datetime, end_datetime).
    Raises ValueError if start >= end.
    """
    end = datetime(2025, 12, 31) if end_date is None else datetime.fromisoformat(end_date)
    start = datetime(2010, 1, 1) if start_date is None else datetime.fromisoformat(start_date)

    if start >= end:
        raise ValueError(f"start_date ({start.date()}) must be before end_date ({end.date()})")

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,   # GIỮ OHLC GỐC
        actions=False,       # KHÔNG thêm cột Dividends/Splits
        group_by="column",
        progress=False,
    )

    # If yf.download returned None, propagate it
    if df is None:
        return None, start, end

    # Normalize columns if yfinance returned a MultiIndex (can happen)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        lvl1 = list(df.columns.get_level_values(1))
        expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        # prefer level that contains expected column names
        if expected.issubset(set(lvl1)):
            df.columns = lvl1
        elif expected.issubset(set(lvl0)):
            df.columns = lvl0
        else:
            # fallback: flatten by joining levels with underscore
            df.columns = ["_".join([str(c) for c in col]).strip() for col in df.columns.values]

    return df, start, end


# ==================== RETURNS ====================

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily return column."""
    df["daily_return"] = df["Adj Close"].pct_change()
    return df


# ==================== TREND INDICATORS ====================

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend indicators (SMA 50, SMA 200)."""
    df["sma50"] = df["Adj Close"].rolling(window=50).mean()
    df["sma200"] = df["Adj Close"].rolling(window=200).mean()
    return df


# ==================== MOMENTUM INDICATORS ====================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    # avoid division by zero producing inf/NaN: replace zeros in avg_loss with NA
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators (RSI, MACD)."""
    # RSI
    df["rsi14"] = compute_rsi(df["Adj Close"], 14)

    # MACD
    ema12 = df["Adj Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


# ==================== VOLATILITY INDICATORS ====================

def add_volatility_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volatility indicators (Bollinger Bands, ATR)."""
    # Bollinger Bands
    df["bb_middle"] = df["Adj Close"].rolling(period).mean()
    bb_std = df["Adj Close"].rolling(period).std()

    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std

    # Avoid potential divide-by-zero by replacing 0 with NA
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, pd.NA)
    df["bb_percent"] = (df["Adj Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, pd.NA)

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=14).mean()

    return df


# ==================== VOLUME INDICATORS ====================

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume (OBV) indicator."""
    # Ensure no NaN in Volume and Close for OBV calculation
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


# ==================== ADDITIONAL FEATURES ====================

def add_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add additional features for better DSS predictions."""
    # Price features
    df["price_change"] = df["Adj Close"].diff()
    # Note: price_change_pct removed - same as daily_return from add_returns()
    
    # High-Low spread
    df["hl_spread"] = df["High"] - df["Low"]
    df["hl_spread_pct"] = (df["High"] - df["Low"]) / df["Close"]
    
    # Open-Close spread
    df["oc_spread"] = df["Close"] - df["Open"]
    df["oc_spread_pct"] = (df["Close"] - df["Open"]) / df["Open"]
    
    # Volume features
    df["volume_sma20"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma20"]
    
    # Price position relative to moving averages
    df["price_vs_sma50"] = (df["Adj Close"] - df["sma50"]) / df["sma50"]
    df["price_vs_sma200"] = (df["Adj Close"] - df["sma200"]) / df["sma200"]
    
    # RSI features
    df["rsi_overbought"] = (df["rsi14"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi14"] < 30).astype(int)
    
    # MACD features
    df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)
    
    # Bollinger Band features
    df["bb_squeeze"] = (df["bb_bandwidth"] < df["bb_bandwidth"].rolling(50).mean() * 0.8).astype(int)
    
    return df


# ==================== MAIN DSS FUNCTION ====================

def generate_nvda_dss_data(ticker: str = "NVDA", 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          output_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Generate comprehensive NVDA data with all technical indicators for DSS.
    
    Args:
        ticker: Stock symbol (default: NVDA)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_file: Output CSV file name
    
    Returns:
        DataFrame with all indicators
    """
    print(f"📊 Fetching {ticker} data...")
    
    # Fetch data
    df, start, end = fetch_prices(ticker, start_date, end_date)
    
    if df is None or df.empty:
        print(f"❌ No data returned for {ticker}")
        return None
    
    # Ensure Date column exists
    if df is not None and not df.empty:
        if "Date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "Date"})
    
    print(f"✅ Data fetched: {len(df)} rows from {start.date()} to {end.date()}")
    
    # Calculate all indicators
    print("📈 Calculating indicators...")
    df = add_returns(df)
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_obv(df)
    df = add_additional_features(df)
    
    # Set default output file if not provided
    if output_file is None:
        output_file = f"{ticker}_dss_features_{datetime.now().strftime('%Y%m%d')}.csv"

    # Save to CSV
    print(f"💾 Saving to {output_file}...")
    out_path = output_file  # now guaranteed to be a str
    df.to_csv(out_path, index=True)
    
    print(f"✅ Done! Saved {len(df)} rows with {len(df.columns)} features")
    
    return df


# ==================== COMMAND LINE INTERFACE ====================

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="NVDA DSS Data Generator - Comprehensive Technical Analysis")
    parser.add_argument("--ticker", "-t", default="NVDA", help="Ticker symbol (default: NVDA)")
    parser.add_argument("--out", "-o", help="Output CSV file path")
    parser.add_argument("--start", help="Start date (ISO YYYY-MM-DD). Defaults to 2010-01-01")
    parser.add_argument("--end", help="End date (ISO YYYY-MM-DD). Defaults to 2025-12-31")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")
    
    args = parser.parse_args()
    
    try:
        # Generate DSS data
        df = generate_nvda_dss_data(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            output_file=args.out
        )
        
        if df is not None and args.summary:
            # Print summary
            print("\n" + "="*60)
            print("  DATA SUMMARY")
            print("="*60)
            # Check if Date is in index or columns
            if 'Date' in df.columns:
                print(f"Date Range: {pd.to_datetime(df['Date']).min().date()} to {pd.to_datetime(df['Date']).max().date()}")
            else:
                print(f"Data Range: {len(df)} rows")
            print(f"Current Price: ${df['Adj Close'].iloc[-1]:.2f}")
            print(f"RSI: {df['rsi14'].iloc[-1]:.1f}")
            print(f"MACD: {'Bullish' if df['macd_bullish'].iloc[-1] else 'Bearish'}")
            print(f"Price vs SMA50: {'Above' if df['price_vs_sma50'].iloc[-1] > 0 else 'Below'}")
            print(f"Price vs SMA200: {'Above' if df['price_vs_sma200'].iloc[-1] > 0 else 'Below'}")
            
    except ValueError as ve:
        print("❌ Input error:", ve)
    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    main()
