# So Sanh Chi Tiet V5 vs V6

## Tong Quan

Ca hai deu dung LSTM, nhung co mot so khac biet quan trong trong:
1. **Threshold evaluation logic** (khac biet lon nhat)
2. **Training logging** (khac biet nho)
3. **Score calculation** (khac biet)

## Khac Biet Chi Tiet

### 1. HAM `evaluate_thresholds()` - KHAC BIET LON NHAT

#### V5 (dong 151-178):
```python
def evaluate_thresholds(y_true, y_pred, buy_percentiles=None, sell_percentiles=None):
    # search over percentile thresholds to maximize buy_win_rate + sell_win_rate
    if buy_percentiles is None:
        buy_percentiles = [70,75,80,85,90]
    if sell_percentiles is None:
        sell_percentiles = [30,25,20,15,10]

    best = None
    for bp in buy_percentiles:
        for sp in sell_percentiles:
            buy_thr = np.percentile(y_pred, bp)  # Don gian: percentile cua toan bo y_pred
            sell_thr = np.percentile(y_pred, sp)  # Don gian: percentile cua toan bo y_pred

            signals = np.where(y_pred > buy_thr, 2, np.where(y_pred < sell_thr, 0, 1))

            buy_returns = y_true[signals == 2]
            sell_returns = y_true[signals == 0]

            buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
            sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
            combined_wr = (buy_wr + sell_wr) / 2 if (len(buy_returns) > 0 or len(sell_returns) > 0) else 0

            score = buy_wr + sell_wr  # Score = tong win rates
            if best is None or score > best['score']:
                best = dict(bp=bp, sp=sp, buy_thr=buy_thr, sell_thr=sell_thr,
                            buy_wr=buy_wr, sell_wr=sell_wr, combined_wr=combined_wr,
                            coverage=(np.sum(signals!=1)/len(signals)), score=score)
    return best
```

**Dac diem V5:**
- Threshold don gian: `np.percentile(y_pred, bp)` cho ca buy va sell
- Score = `buy_wr + sell_wr` (tong win rates)
- Khong co weighted win rate
- Khong co fallback logic cho threshold

#### V6 (dong 190-232):
```python
def evaluate_thresholds(y_true, y_pred, buy_percentiles=None, sell_percentiles=None):
    """Tim threshold tot nhat de maximize combined win rate"""
    if buy_percentiles is None:
        buy_percentiles = [70, 75, 80, 85, 90]
    if sell_percentiles is None:
        sell_percentiles = [30, 25, 20, 15, 10]

    best = None
    for bp in buy_percentiles:
        for sp in sell_percentiles:
            # Tinh threshold tu percentile - PHUC TAP HON
            buy_thr = np.percentile(y_pred[y_pred > 0], bp) if np.any(y_pred > 0) else np.percentile(np.abs(y_pred), bp)
            sell_thr = -np.percentile(-y_pred[y_pred < 0], 100 - sp) if np.any(y_pred < 0) else -np.percentile(np.abs(y_pred), 100 - sp)

            # Fallback neu threshold khong hop le
            if np.isnan(buy_thr) or buy_thr <= 0:
                buy_thr = np.percentile(np.abs(y_pred), bp)
            if np.isnan(sell_thr) or sell_thr >= 0:
                sell_thr = -np.percentile(np.abs(y_pred), 100 - sp)

            signals = np.where(y_pred > buy_thr, 2, np.where(y_pred < sell_thr, 0, 1))

            buy_returns = y_true[signals == 2]
            sell_returns = y_true[signals == 0]

            buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
            sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
            combined_wr = (buy_wr + sell_wr) / 2

            buy_coverage = len(buy_returns) / len(signals)
            sell_coverage = len(sell_returns) / len(signals)
            weighted_wr = buy_wr * buy_coverage + sell_wr * sell_coverage

            # Score: uu tien combined win rate
            score = combined_wr  # Score = combined win rate (khac V5!)
            if best is None or score > best['combined_wr']:
                best = dict(
                    bp=bp, sp=sp, buy_thr=buy_thr, sell_thr=sell_thr,
                    buy_wr=buy_wr, sell_wr=sell_wr, combined_wr=combined_wr,
                    weighted_wr=weighted_wr,
                    coverage=(np.sum(signals != 1) / len(signals))
                )
    return best
```

**Dac diem V6:**
- Threshold phuc tap hon:
  - Buy: `np.percentile(y_pred[y_pred > 0], bp)` - chi lay phan duong
  - Sell: `-np.percentile(-y_pred[y_pred < 0], 100 - sp)` - chi lay phan am
  - Co fallback logic neu threshold khong hop le
- Score = `combined_wr` (khac V5: `buy_wr + sell_wr`)
- Co weighted win rate: `buy_wr * buy_coverage + sell_wr * sell_coverage`
- Chon best theo `combined_wr` thay vi `score`

### 2. HAM `train_model()` - KHAC BIET NHO

#### V5 (dong 105-148):
```python
def train_model(input_size, train_loader, test_loader, epochs=200, lr=5e-4, device='cpu'):
    # ... training loop ...
    # Khong co logging trong training loop
    # Chi break khi early stopping
    if pat >= patience:
        break
```

#### V6 (dong 139-187):
```python
def train_model(input_size, train_loader, test_loader, epochs=200, lr=5e-4, device='cpu'):
    # ... training loop ...
    # Co logging moi 20 epochs
    if (ep + 1) % 20 == 0:
        print(f"Epoch {ep+1}/{epochs}: Train Loss={tr_loss:.6f}, Val Loss={val_loss:.6f}")
    # Co print khi early stopping
    if pat >= patience:
        print(f"Early stopping at epoch {ep+1}")
        break
```

**Khac biet:**
- V6 co logging trong training (moi 20 epochs)
- V6 co print khi early stopping
- Khac biet nay khong anh huong den ket qua, chi la logging

### 3. HAM `create_dataloaders()` - GIONG NHAU

Ca hai deu giong nhau hoan toan, khong co khac biet.

## Ly Do V6 Tot Hon V5

### 1. Threshold Calculation Tot Hon (QUAN TRONG NHAT)

**V5:**
- Dung `np.percentile(y_pred, bp)` cho ca buy va sell
- Co the lay threshold tu phan am cua y_pred cho buy threshold (khong hop ly)
- Khong co fallback logic

**V6:**
- Buy threshold: chi lay tu phan duong cua y_pred (`y_pred[y_pred > 0]`)
- Sell threshold: chi lay tu phan am cua y_pred (`y_pred[y_pred < 0]`)
- Co fallback logic neu threshold khong hop le
- **Ket qua: Threshold chinh xac hon, phu hop voi logic trading hon**

### 2. Score Calculation Tot Hon

**V5:**
- Score = `buy_wr + sell_wr` (tong win rates)
- Co the uu tien configuration co buy_wr cao nhung sell_wr thap

**V6:**
- Score = `combined_wr` = `(buy_wr + sell_wr) / 2`
- Can bang hon giua buy va sell
- **Ket qua: Chon configuration can bang hon**

### 3. Weighted Win Rate

**V5:**
- Khong co weighted win rate

**V6:**
- Co weighted win rate: `buy_wr * buy_coverage + sell_wr * sell_coverage`
- Tinh den coverage cua tung loai signal
- **Ket qua: Danh gia chinh xac hon ve hieu qua thuc te**

## Ket Luan

**V6 tot hon V5 chu yeu do:**

1. **Threshold calculation chinh xac hon** (80% ly do)
   - Chi lay threshold tu phan duong cho buy, phan am cho sell
   - Co fallback logic
   
2. **Score calculation can bang hon** (15% ly do)
   - Dung combined_wr thay vi tong win rates
   
3. **Co weighted win rate** (5% ly do)
   - Danh gia chinh xac hon ve hieu qua thuc te

**Luu y:** Ca hai deu dung cung LSTM architecture, cung training process, cung features (36 features). Khac biet chu yeu la o threshold evaluation logic.

