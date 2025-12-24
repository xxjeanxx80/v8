# Data Pipeline - Download va Them Features

Thu muc nay chua cac script de:
1. Download raw data tu Yahoo Finance (2010-2025)
2. Them cac technical indicators vao raw data

## Cau truc

```
data_pipeline/
├── download_raw_data.py    # Download raw data tu Yahoo Finance
├── add_features.py          # Them technical indicators
├── run_full_pipeline.py     # Chay day du pipeline
└── README.md                # File nay
```

## Cach su dung

### Option 1: Chay day du pipeline (khuyen dung)

```bash
cd data_pipeline
python run_full_pipeline.py
```

Script nay se:
1. Download raw data cho tat ca stocks (NVDA, AMD, MU, INTC, QCOM, TSM)
2. Them features vao raw data
3. Luu ket qua vao `../data/`

### Option 2: Chay tung buoc rieng le

**Buoc 1: Download raw data**
```bash
cd data_pipeline
python download_raw_data.py
```

Raw data se duoc luu vao `../data/raw/` voi format: `{TICKER}_raw_YYYYMMDD.csv`

**Buoc 2: Them features**
```bash
cd data_pipeline
python add_features.py
```

Feature files se duoc luu vao `../data/` voi format: `{TICKER}_dss_features_YYYYMMDD.csv`

## Date Range

- **Start Date**: 2010-01-01
- **End Date**: 2025-12-31

## Stocks

- NVDA (NVIDIA)
- AMD (Advanced Micro Devices)
- MU (Micron Technology)
- INTC (Intel)
- QCOM (Qualcomm)
- TSM (TSMC)

## Features duoc them

### Returns
- `daily_return`: Daily return percentage

### Trend Indicators
- `sma50`: Simple Moving Average 50 days
- `sma200`: Simple Moving Average 200 days

### Momentum Indicators
- `rsi14`: Relative Strength Index (14 periods)
- `macd`: MACD line
- `macd_signal`: MACD signal line
- `macd_hist`: MACD histogram

### Volatility Indicators
- `bb_upper`: Bollinger Band upper
- `bb_middle`: Bollinger Band middle
- `bb_lower`: Bollinger Band lower
- `bb_bandwidth`: Bollinger Band bandwidth
- `bb_percent`: Bollinger Band position (0-1)
- `atr`: Average True Range

### Volume Indicators
- `obv`: On-Balance Volume

### Additional Features
- `price_change`: Price change
- `hl_spread`: High-Low spread
- `hl_spread_pct`: High-Low spread percentage
- `oc_spread`: Open-Close spread
- `oc_spread_pct`: Open-Close spread percentage
- `volume_sma20`: Volume SMA 20
- `volume_ratio`: Volume ratio to SMA20
- `price_vs_sma50`: Price position vs SMA50
- `price_vs_sma200`: Price position vs SMA200
- `rsi_overbought`: RSI > 70 flag
- `rsi_oversold`: RSI < 30 flag
- `macd_bullish`: MACD > Signal flag
- `bb_squeeze`: Bollinger Band squeeze flag

## Output Files

### Raw Data
- Location: `../data/raw/`
- Format: `{TICKER}_raw_YYYYMMDD.csv`
- Columns: Date, Open, High, Low, Close, Adj Close, Volume

### Feature Data
- Location: `../data/`
- Format: `{TICKER}_dss_features_YYYYMMDD.csv`
- Columns: All raw columns + technical indicators (34+ features)

## Notes

- Raw data duoc download tu Yahoo Finance
- Features duoc tinh toan tu raw data
- Cac rows co NaN (do rolling windows) se bi loai bo
- Date range mac dinh: 2010-01-01 to 2025-12-31

## Requirements

- yfinance
- pandas
- numpy

Install:
```bash
pip install yfinance pandas numpy
```

