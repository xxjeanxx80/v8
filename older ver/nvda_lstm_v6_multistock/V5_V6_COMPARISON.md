# So Sanh Ket Qua V5 vs V6

## Tong Quan

- **V5**: Sử dụng ~18-25 features từ `optimized_feature_config.csv`
- **V6**: Sử dụng 10 features tối ưu từ quick test (remove_top7_vif)

## Ket Qua V6 (Da Chay)

Từ terminal output, v6 đã chạy với kết quả:

### Metrics V6:
| Metric | Value | Expected (Quick Test) | Status |
|--------|-------|----------------------|--------|
| **Combined Win Rate** | 22.5% | 69.65% | ❌ Thấp hơn nhiều |
| **Buy Win Rate** | 0.0% | 93.94% | ❌ Rất thấp |
| **Sell Win Rate** | 45.0% | 45.36% | ✅ Tương đương |
| **RMSE** | 0.0491 | 0.0402 | ⚠️ Cao hơn |
| **MAE** | 0.0394 | - | - |
| **Coverage** | 15.4% | - | - |
| **Số features** | 10 | 10 | ✅ Đúng |

### Features V6 (10 features):
1. oc_spread_pct
2. oc_spread
3. volume_sma20
4. bb_bandwidth
5. macd_bullish
6. hl_spread_pct
7. rsi_overbought
8. volume_ratio
9. bb_squeeze
10. rsi_oversold

### Threshold V6:
- **Buy percentile**: 70 -> threshold 0.0197
- **Sell percentile**: 15 -> threshold -0.0218

## Ket Qua V5 (Can Chay)

V5 chưa có artifact. Cần chạy v5 để so sánh.

### Expected V5:
- Sử dụng ~18-25 features (từ `optimized_feature_config.csv`)
- Metrics sẽ được hiển thị sau khi chạy

## Phan Tich Van De V6

### 1. Buy Win Rate = 0%
**Nguyên nhân có thể:**
- Threshold quá cao (percentile 70 -> 0.0197)
- Model không học được pattern buy với 10 features
- Features không đủ để predict buy signals
- Training chưa đủ (early stopping ở epoch 40)

**Giải pháp:**
- Giảm buy percentile xuống 60-65
- Tăng epochs hoặc giảm patience
- Kiểm tra distribution của predictions

### 2. Combined WR thấp (22.5% vs 69.65%)
**Nguyên nhân có thể:**
- Dataset khác nhau giữa quick test và v6
- Training parameters khác nhau:
  - Quick test: epochs=50, patience=10
  - V6: epochs=200, patience=20 (nhưng early stop ở 40)
- Model chưa train đủ

**Giải pháp:**
- Đồng bộ training parameters với quick test
- Kiểm tra data split có giống nhau không

### 3. RMSE cao hơn expected
- 0.0491 vs 0.0402 (cao hơn ~22%)
- Có thể do model chưa train đủ hoặc features không đủ

## Khuyen Nghi

### Ngay lap tuc:
1. **Chạy v5** để có baseline so sánh
2. **Điều chỉnh v6**:
   - Giảm buy percentile xuống 60-65
   - Đồng bộ training parameters với quick test (epochs=50, patience=10)
   - Kiểm tra distribution của predictions

### Sau khi co ket qua v5:
1. So sánh chi tiết metrics
2. Xác định features nào quan trọng nhất
3. Quyết định giữ v5 hay v6

## Cach Chay So Sanh

### Cach 1: Chay v5 roi so sanh
```powershell
cd "C:\Users\xxjea\Downloads\DSS DATA\nvda_lstm_v6_multistock"
.\run_v5_then_compare.ps1
```

### Cach 2: Chay thu cong
```powershell
# Chay v5
cd "C:\Users\xxjea\Downloads\DSS DATA\nvda_lstm_v5_multistock"
..\.venv311\Scripts\Activate.ps1
python nvda_lstm_v5_multistock.py --epochs 50

# So sanh
cd "..\nvda_lstm_v6_multistock"
python compare_v5_v6.py
```

## Ket Qua So Sanh Se Co

Sau khi chạy v5, script `compare_v5_v6.py` sẽ tạo file `v5_v6_comparison.csv` với:
- So sánh số lượng features
- So sánh từng metric (RMSE, MAE, Win Rates)
- Features được thêm/bớt
- Đánh giá tổng thể

