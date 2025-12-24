# NVDA LSTM v7 - Advanced Trading Strategy

## Tong Quan

V7 la phien ban nang cap cua V6 voi cac cai tien quan trong:

## V7.1 - No-Leakage Strategy

V7.1 la phien ban cai tien cua V7 voi muc dich tranh data leakage:

**Khac biet chinh:**
- **Pretrain**: 2015-2020 (5 nam) thay vi 2015-2025 (10 nam) - TACH BIET voi test
- **Fine-tune**: 2021-2024 (4 nam) thay vi 2021-2025 (4 nam) - TACH BIET voi test
- **Test**: 2025 (1 nam) - OUT-OF-SAMPLE thuc su (khong co trong pretrain/finetune)

**Muc dich:**
- Giam overfit: Tranh data leakage (V7 co the leak vi pretrain 2015-2025 bao gom test period 2025)
- Tang generalization: Pretrain tren period khac hoan toan voi test period
- Out-of-sample thuc su: Test period 2025 khong xuat hien trong pretrain/finetune

**Cach su dung V7.1:**
```bash
cd nvda_lstm_v7_multistock
python nvda_lstm_v7_1_multistock.py
```

Hoac voi GPU:
```powershell
.\run_v7_1_gpu.ps1
```

**So sanh V7 vs V7.1:**
```bash
python compare_v7_v7_1.py
```

## V7 - Advanced Trading Strategy

V7 la phien ban nang cap cua V6 voi cac cai tien quan trong:

1. **V7.1: Expectancy Proxy Scoring** - Uu tien signals co loi nhuan thuc te thay vi chi win rate
2. **V7.2: Center y_pred** - Threshold on dinh hon qua cac regime khac nhau
3. **V7.3: Asymmetric Scoring** - Uu tien BUY cho NVDA (70% buy, 30% sell)
4. **Overfitting Detection** - Walk-forward analysis de phat hien overfitting
5. **False Signal Filtering** - Loai bo signals co win rate thap, return am, expectancy < 0
6. **Hybrid Trade/No-Trade Logic** - Chi trade khi confidence cao va co expectancy
7. **Profitability Verification** - Kiem tra lai/lo thuc te (total return, profit factor, Sharpe, max drawdown)

## Cai Tien So Voi V6

### V7.1: Expectancy Proxy Scoring

Thay vi chi dung win rate, V7 tinh expectancy:

```python
buy_expectancy = buy_wr * avg_buy_win - (1 - buy_wr) * abs(avg_buy_loss)
sell_expectancy = sell_wr * avg_sell_win - (1 - sell_wr) * abs(avg_sell_loss)

score = buy_weight * buy_expectancy * buy_coverage + sell_weight * sell_expectancy * sell_coverage
```

**Loi ich**: Uu tien signals co loi nhuan thuc te thay vi chi win rate cao nhung return thap.

### V7.2: Center y_pred

Center predictions truoc khi tinh threshold:

```python
y_pred_mean = np.mean(y_pred)
y_pred_centered = y_pred - y_pred_mean
```

**Loi ich**: Threshold on dinh hon qua cac regime khac nhau (bull/bear/sideways).

### V7.3: Asymmetric Scoring

Uu tien BUY cho NVDA (default: 70% buy, 30% sell):

```python
buy_weight = 0.7
sell_weight = 0.3
```

**Loi ich**: Phu hop voi upside drift cua NVDA.

### False Signal Detection

Loai bo signals khong dat tieu chuan:
- Win rate < 50%
- Average return am
- Expectancy < 0.01

**Loi ich**: Chi giu lai signals chat luong cao.

### Hybrid Trade/No-Trade Logic

Chi trade khi:
- Confidence cao: `|y_pred_centered| > confidence_threshold` (top 40%)
- Co expectancy: `expectancy > min_expectancy` (0.01)

**Loi ich**: Tranh tin hieu gia, chi trade khi co xac suat thanh cong cao.

### Overfitting Detection

Walk-forward analysis:
- Chia test set thanh n windows
- Danh gia model tren tung window
- Check: test_loss >> train_loss hoac metrics giam dan

**Loi ich**: Phat hien overfitting som.

### Profitability Verification

Tinh cac metrics thuc te:
- Total Return
- Profit Factor
- Sharpe Ratio
- Max Drawdown

**Loi ich**: Xac nhan strategy co lai thuc te hay khong.

## Cach Su Dung

### Basic Usage

```bash
cd nvda_lstm_v7_multistock
python nvda_lstm_v7_multistock.py
```

### Voi GPU

```powershell
.\run_v7_gpu.ps1
```

### Custom Configuration

```bash
python nvda_lstm_v7_multistock.py \
    --buy_weight 0.7 \
    --sell_weight 0.3 \
    --min_wr 0.50 \
    --min_expectancy 0.01 \
    --confidence_pct 60 \
    --walk_forward_windows 3 \
    --epochs 200
```

## Configuration Parameters

- `--buy_weight`: Weight cho BUY signals (default: 0.7)
- `--sell_weight`: Weight cho SELL signals (default: 0.3)
- `--min_wr`: Min win rate de khong bi coi la false signal (default: 0.50)
- `--min_expectancy`: Min expectancy de trade (default: 0.01)
- `--confidence_pct`: Confidence threshold percentile (default: 60 = top 40%)
- `--walk_forward_windows`: So windows cho walk-forward analysis (default: 3)

## Output Metrics

### Trading Metrics
- Buy/Sell Win Rate
- Combined Win Rate
- Buy/Sell Expectancy
- Coverage (Buy + Sell)
- Expectancy Score

### Profitability Metrics
- Total Return
- Profit Factor
- Sharpe Ratio
- Max Drawdown
- Is Profitable

### Overfitting Metrics
- Overfitting Score (Test Loss / Train Loss)
- Is Overfitting (True/False)
- Test Metrics Std (consistency)
- Metrics Decreasing (True/False)
- Window Details

### False Signal Detection
- Buy is False (True/False)
- Sell is False (True/False)
- So luong signals bi loai bo

## Expected Improvements

1. **Better Signal Quality**: False signal filtering loai bo signals kem
2. **Stable Thresholds**: Center y_pred giup threshold on dinh hon
3. **Profitability Focus**: Expectancy proxy uu tien signals co loi nhuan thuc te
4. **NVDA-Optimized**: Asymmetric scoring phu hop voi upside drift cua NVDA
5. **Overfitting Detection**: Walk-forward giup phat hien overfitting som
6. **Trade/No-Trade Logic**: Hybrid logic tranh tin hieu gia

## So Sanh Voi V6

| Feature | V6 | V7 |
|---------|----|----|
| Scoring | Combined Win Rate | Expectancy Proxy |
| Threshold | Direct percentile | Center y_pred first |
| Weights | Equal (50/50) | Asymmetric (70/30) |
| False Signals | No filtering | Filter by WR/Return/Expectancy |
| Trade Logic | Simple threshold | Hybrid (confidence + expectancy) |
| Overfitting Check | No | Walk-forward analysis |
| Profitability | Basic | Full metrics |

## Notes

- V7 su dung TAT CA features co san (khong bo feature nao)
- LSTM can correlation de tang kha nang vao lenh
- VIF lam mat tri nho cua LSTM, khong nen bo features theo VIF

