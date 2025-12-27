# NVDA LSTM v6 - Optimized với 10 Features

Version này sử dụng **10 features tốt nhất** từ quick test (`remove_top7_vif`).

## Kết quả từ Quick Test

**Best Configuration: `remove_top7_vif`**
- **Combined Win Rate**: 69.65% (cao nhất)
- **Buy Win Rate**: 93.94% (rất cao!)
- **Sell Win Rate**: 45.36%
- **RMSE**: 0.0402 (tốt hơn baseline 22.4%)
- **Average VIF**: 3.20 (giảm từ hàng nghìn xuống 3.2)
- **Số lượng features**: 10 features (giảm từ 25 xuống 10, -60%)

## Features được sử dụng (10 features)

Từ baseline 17 features, loại bỏ 7 features có VIF cao nhất:

**Loại bỏ:**
1. `macd_hist` (VIF: 142,971,416,741,920.5)
2. `macd_signal` (VIF: 25,957,346,555,449.5)
3. `macd` (VIF: 999.0)
4. `rsi14` (VIF: 15.88)
5. `bb_percent` (VIF: 12.43)
6. `obv` (VIF: 9.70)
7. `atr` (VIF: 7.63)

**Còn lại (10 features):**
1. `macd_bullish`
2. `bb_bandwidth`
3. `volume_ratio`
4. `volume_sma20`
5. `daily_return`
6. `price_change`
7. `return_3d`
8. `return_5d`
9. `return_10d`
10. `return_20d`
11. `hl_spread_pct`
12. `oc_spread`
13. `oc_spread_pct`
14. `bb_squeeze`
15. `rsi_overbought`
16. `rsi_oversold`
17. `sox_beta`
18. `sox_correlation`

**Lưu ý**: Nếu dataset có đủ features, sẽ dùng tất cả 18 features trên. Nếu không, sẽ filter theo available features.

## So sánh với v5

| Metric | v5 (25 features) | v6 (10 features) | Thay đổi |
|--------|------------------|------------------|----------|
| Combined WR | ~67.62% | 69.65% (expected) | +2.03% |
| Buy WR | ~90.91% | 93.94% (expected) | +3.03% |
| Sell WR | ~44.33% | 45.36% (expected) | +1.03% |
| RMSE | ~0.0518 | 0.0402 (expected) | -22.4% |
| Avg VIF | Rất cao | 3.20 | -99.99% |
| Features | 25 | 10 | -60% |

## Cách sử dụng

```bash
# Chạy với default parameters
python nvda_lstm_v6_multistock.py

# Chạy với custom parameters
python nvda_lstm_v6_multistock.py --epochs 200 --batch_size 64 --data_dir ../data
```

## Output

- `best_v6_reg.pth`: Model weights
- `nvda_lstm_v6_artifact.pth`: Model + features + metrics

## Mục đích

Test lại với 10 features để:
1. Xác nhận kết quả từ quick test
2. So sánh với v5 (25 features)
3. Đánh giá xem việc giảm features có ảnh hưởng gì không

