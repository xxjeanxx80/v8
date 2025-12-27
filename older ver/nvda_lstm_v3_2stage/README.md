# NVDA LSTM - Version 3: 2-Stage Filtering

## Overview
Advanced 2-stage system with separate models for trade detection and direction prediction.

## Architecture
- **Stage 1 (Filter)**: Volume + Volatility features → Trade/NoTrade
- **Stage 2 (Direction)**: Price + Momentum features → Buy/Sell
- **Stage 3 (Entry Gate)**: Rule-based breakout confirmation

## Features
- Separate feature groups for each stage
- Volume spike detection
- Volatility regime filtering
- Breakout confirmation signals
- 3-stage inference pipeline

## Files
- `nvda_lstm_2stage.py` - Complete 2-stage implementation

## Key Results
- Stage 1 Accuracy: 93.98%
- Stage 2 Accuracy: 100% (on filtered trades)
- Coverage: 6% (very selective)
- Entry Gate: 100% filter rate (too strict)

## Improvements from v2
- More selective trading
- Separated concerns (filter vs direction)
- Rule-based entry confirmation

## Issues Found
- Entry gate too strict (0% coverage)
- Need more data for reliable Stage 2 training

## Next Steps
→ Multi-stock transfer learning in v4
