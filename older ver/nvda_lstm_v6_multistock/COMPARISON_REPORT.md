# So Sanh Ket Qua V5 vs V6

## Ket Qua V6 (Da Chay)

Từ terminal output, v6 đã chạy với kết quả:

### Metrics V6:
- **Combined Win Rate**: 22.5% (thấp hơn expected 69.65%)
- **Buy Win Rate**: 0.0% (rất thấp, expected 93.94%)
- **Sell Win Rate**: 45.0% (tốt hơn expected 45.36%)
- **RMSE**: 0.0491 (cao hơn expected 0.0402)
- **MAE**: 0.0394
- **Coverage**: 15.4%
- **Số features**: 10 features

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

## Ket Qua V5 (Can Chay)

V5 chưa có artifact. Cần chạy v5 để so sánh.

### Expected V5 (từ code):
- Sử dụng ~18-25 features (từ optimized_feature_config.csv)
- Metrics sẽ được hiển thị sau khi chạy

## Nhan Xet

### Vấn đề với V6:
1. **Buy Win Rate = 0%**: Rất bất thường, có thể do:
   - Threshold quá cao
   - Model không học được pattern buy
   - Features không đủ để predict buy signals

2. **Combined WR thấp (22.5%)**: Thấp hơn nhiều so với expected (69.65%)
   - Có thể do dataset khác nhau
   - Hoặc do training parameters khác nhau (epochs, learning rate)

3. **RMSE cao hơn expected**: 0.0491 vs 0.0402
   - Có thể do model chưa train đủ
   - Hoặc do features không đủ

### So sánh với Quick Test:
- Quick test sử dụng epochs=50, patience=10 (nhanh hơn)
- V6 sử dụng epochs=200, patience=20 (đầy đủ hơn)
- Có thể cần điều chỉnh training parameters

## Khuyen Nghi

1. **Chạy v5** để có baseline so sánh
2. **Điều chỉnh v6**:
   - Giảm epochs xuống 50 (như quick test)
   - Giảm patience xuống 10
   - Kiểm tra threshold search
3. **Kiểm tra data**: Đảm bảo dataset giống nhau giữa quick test và v6

## Cach Chay So Sanh

```powershell
# Chay v5 truoc (neu chua co artifact)
cd "C:\Users\xxjea\Downloads\DSS DATA\nvda_lstm_v5_multistock"
..\.venv311\Scripts\Activate.ps1
python nvda_lstm_v5_multistock.py --epochs 50

# Sau do chay so sanh
cd "C:\Users\xxjea\Downloads\DSS DATA\nvda_lstm_v6_multistock"
python compare_v5_v6.py
```

