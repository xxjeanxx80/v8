# NVDA LSTM v8 - Final Version

## Tong quan

V8 la version cuoi cung thua huong toan bo logic tu V7.1 voi cac thay doi ve date ranges de tang kha nang danh gia va giam overfitting.

## Date Ranges Strategy (V8)

- **Pretrain**: 2015-2020 (5 nam) - all stocks, giu nguyen nhu V7.1
- **Fine-tune**: 2021-2023 (3 nam) - all stocks, freeze encoder (thay doi tu 2021-2024)
- **Validation**: 2022-2023-NVDA (2 nam) - NVDA only, cho threshold optimization (thay doi tu 2023-2024)
- **Test**: 2024-2025 (2 nam) - NVDA only, out-of-sample (thay doi tu chi 2025)

## Khac biet voi V7.1

1. **Fine-tune period ngan hon**: 2021-2023 (3 nam) thay vi 2021-2024 (4 nam)
2. **Validation period som hon**: 2022-2023 thay vi 2023-2024
3. **Test period dai hon**: 2024-2025 (2 nam) thay vi chi 2025 (1 nam)
4. **More out-of-sample data**: Test period khong overlap voi fine-tune

## Files trong folder

### 1. Main Script
- `nvda_lstm_v8_multistock.py`: Script chinh de train va test model V8

### 2. Comparison Script
- `compare_v5_v6_v7.py`: So sanh ket qua cua V5, V6, V7

### 3. Trading Analysis
- `trading_analysis_2024_2025/analyze_trading_2024_2025.py`: Phan tich ket qua trading cho 2024-2025

## Cach su dung

### Chay main script

```bash
python nvda_lstm_v8_multistock.py --data_dir ../data
```

### So sanh versions

```bash
python compare_v5_v6_v7.py
```

### Phan tich trading

```bash
python trading_analysis_2024_2025/analyze_trading_2024_2025.py --artifact nvda_lstm_v8_artifact.pth
```

## Output

- `nvda_lstm_v8_artifact.pth`: Artifact chua model state, metrics, thresholds
- `v5_v6_v7_comparison.csv`: Bang so sanh metrics giua V5, V6, V7
- `V5_V6_V7_COMPARISON.md`: Markdown report so sanh
- `trading_analysis_2024_2025/trading_results_*.csv`: Ket qua phan tich trading
- `trading_analysis_2024_2025/trading_analysis_2024_2025.png`: Bieu do ket qua

## Thua huong tu V7.1

V8 thua huong toan bo cac cai tien tu V7.1:
- Expectancy proxy scoring
- Center y_pred truoc threshold
- Asymmetric scoring cho NVDA (uu tien BUY: 70/30)
- Overfitting detection
- False signal filtering
- Hybrid trade/no-trade logic
- Profitability verification

## Notes

- Tat ca logic tu V7.1 duoc giu nguyen
- Chi thay doi date ranges va mo rong test period
- Comparison script ho tro so sanh 3 versions cung luc
- Trading analysis ho tro ca tong hop va rieng tung nam

