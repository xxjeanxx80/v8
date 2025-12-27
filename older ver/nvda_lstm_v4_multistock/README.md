# NVDA LSTM - Version 4: Multi-Stock Transfer Learning

## Overview
Complete multi-stock system with transfer learning and best practices for cross-stock generalization.

## Architecture
- **Train on**: NVDA + AMD + MU + INTC + QCOM (5 semiconductor stocks)
- **Test on**: NVDA only (out-of-sample)
- **Features**: 35 relative features (no absolute prices)
- **Labels**: Per-stock quantile thresholds

## Key Features
✅ Per-stock quantile labels (adaptive thresholds)
✅ Sector features (SOX beta, correlation)
✅ Only relative features (returns, ratios, positions)
✅ Proper transfer learning split
✅ One-hot stock encoding

## Files
- `nvda_lstm_multistock_complete.py` - Complete implementation with best practices
- `nvda_lstm_multistock.py` - Initial multi-stock version
- `download_stocks.py` - Script to download real stock data
- `test_multistock.py` - Create synthetic data for testing
- Stock data files: AMD, MU, INTC, QCOM features

## Key Results
- **Data**: 3905 rows (5x more than single stock)
- **RMSE**: 0.0401 (improved accuracy)
- **Win Rate BUY**: 91.67% at 2% threshold
- **Coverage**: 19.05% (more selective but higher precision)

## Best Practices Implemented
1. No absolute price features - only relative metrics
2. Per-stock quantile calculations
3. Sector-wide features for shared patterns
4. Proper train/test split avoiding look-ahead bias

## How to Use
```bash
# Download real data (optional)
python download_stocks.py

# Generate features for each stock
# Run main model
python nvda_lstm_multistock_complete.py
```

## Evolution Summary
- v1: Basic PyTorch LSTM
- v2: 5-day prediction with enhanced features
- v3: 2-stage filtering system
- v4: Multi-stock transfer learning with best practices
