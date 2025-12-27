# NVDA LSTM - Version 2: 5-Day Prediction

## Overview
Extended version with 5-day ahead prediction and enhanced features.

## Features
- 5-day future return prediction
- Enhanced feature engineering
- Regression + Classification approach
- Balanced quantile-based labels
- Reduced sequence length (30 days) for more samples

## Files
- `nvd_lstm_5day.py` - Main 5-day implementation
- `nvda_lstm_enhanced.py` - Enhanced version with more features

## Key Results
- Win Rate BUY: 87.5% at 2% threshold
- Coverage: 47.3% (70/148 trades)
- RMSE: 0.0446

## Improvements from v1
- Longer prediction horizon (5 days)
- Better feature set
- More balanced approach

## Next Steps
→ 2-stage filtering system in v3
