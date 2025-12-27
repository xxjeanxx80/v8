# Feature Testing Framework

This folder contains tools to scientifically test and optimize the feature set for the NVDA LSTM trading model.

## Files

- `feature_analysis.py` - Correlation analysis, VIF calculation, and feature grouping
- `ablation_test.py` - Ablation testing to measure feature group importance
- `README.md` - This documentation

## Methodology

### 1. Correlation Analysis
- Identifies redundant features with |correlation| > 0.9
- Generates heatmap visualization
- Recommends removing one of each highly correlated pair

### 2. Variance Inflation Factor (VIF)
- Detects multicollinearity that pairwise correlation misses
- VIF > 5 indicates potential redundancy
- Helps identify features that explain the same variance

### 3. Ablation Testing
- Tests model performance with different feature groups removed:
  - Trend features (price vs moving averages)
  - Momentum features (RSI, MACD)
  - Volatility features (ATR, Bollinger Bands)
  - Volume features (volume ratio, OBV)
  - Return features (various timeframe returns)
  - Structure features (price spreads)
  - Reduced set (14-16 core features)

### 4. Statistical Significance
- Compares full vs reduced feature set
- Measures performance impact of feature removal
- Provides data-driven recommendations

## Usage

### Step 1: Feature Correlation Analysis
```bash
cd test_feature
python feature_analysis.py
```
Outputs:
- `correlation_heatmap.png` - Visual correlation matrix
- `feature_analysis_summary.csv` - Summary statistics
- Console output with redundant feature pairs

### Step 2: Ablation Testing
```bash
python ablation_test.py
```
Outputs:
- `ablation_results.csv` - Performance metrics for each feature group
- `ablation_results.png` - Visual comparison of results
- Console output with recommendations

## Expected Results

Based on domain knowledge, we expect to reduce from 34 features to 14-16 core features without significant performance loss:

### Core Features (14-16):
**Trend:**
- price_vs_sma50
- price_vs_sma200

**Momentum:**
- rsi14
- macd
- macd_bullish

**Volatility:**
- atr
- bb_bandwidth
- bb_percent

**Volume:**
- volume_ratio
- obv

**Structure:**
- hl_spread_pct

**Returns (lagged):**
- daily_return
- price_change

**Position:**
- bb_squeeze

## Decision Criteria

- Keep reduced set if RMSE increase < 10%
- Keep reduced set if win rate drop < 5%
- Consider feature efficiency (win rate per feature)
- Prioritize interpretability and speed

## Next Steps

1. Run analysis to identify redundant features
2. Test ablation to measure group importance
3. Implement optimized feature set in production model
4. Document final feature selection rationale
