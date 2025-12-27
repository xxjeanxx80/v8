# GIAI THICH CACH TINH BUY WIN RATE VA SELL WIN RATE

## 1. BUY WIN RATE (Tỷ lệ thắng khi mua)

### Logic:
- **BUY signal** (Signal = 2): Model dự đoán giá sẽ TĂNG → Quyết định MUA
- **BUY WIN**: Khi thực tế giá TĂNG (Actual_Return > 0) → Lãi → WIN
- **BUY LOSS**: Khi thực tế giá GIẢM hoặc KHÔNG ĐỔI (Actual_Return <= 0) → Lỗ → LOSS

### Công thức:
```
Buy Win Rate = Số BUY signals có Actual_Return > 0 / Tổng số BUY signals
```

### Ví dụ cụ thể:

| Ngày | Signal | Predicted_Return | Actual_Return | Kết quả |
|------|--------|-----------------|---------------|---------|
| 2024-02-14 | BUY (2) | 0.0077 | 0.0628 | **WIN** (vì 0.0628 > 0) |
| 2024-03-07 | BUY (2) | 0.0060 | -0.0510 | **LOSS** (vì -0.0510 <= 0) |
| 2024-05-01 | BUY (2) | 0.0066 | 0.0888 | **WIN** (vì 0.0888 > 0) |

**Ví dụ tính toán:**
- Tổng số BUY signals: 99
- Số BUY WIN (Actual_Return > 0): 73
- **Buy Win Rate = 73/99 = 73.7%**

---

## 2. SELL WIN RATE (Tỷ lệ thắng khi bán)

### Logic:
- **SELL signal** (Signal = 0): Model dự đoán giá sẽ GIẢM → Quyết định BÁN (SHORT)
- **SELL WIN**: Khi thực tế giá GIẢM (Actual_Return < 0) → Lãi → WIN
  - Vì SELL = SHORT position, nên giá giảm → lãi
- **SELL LOSS**: Khi thực tế giá TĂNG hoặc KHÔNG ĐỔI (Actual_Return >= 0) → Lỗ → LOSS
  - Vì SELL = SHORT position, nên giá tăng → lỗ

### Công thức:
```
Sell Win Rate = Số SELL signals có Actual_Return < 0 / Tổng số SELL signals
```

### Ví dụ cụ thể:

| Ngày | Signal | Predicted_Return | Actual_Return | Kết quả |
|------|--------|-----------------|---------------|---------|
| 2024-07-12 | SELL (0) | -0.0104 | -0.0875 | **WIN** (vì -0.0875 < 0, giá giảm → lãi) |
| 2024-05-14 | SELL (0) | -0.0099 | 0.0441 | **LOSS** (vì 0.0441 >= 0, giá tăng → lỗ) |
| 2024-11-19 | SELL (0) | -0.0094 | -0.0686 | **WIN** (vì -0.0686 < 0, giá giảm → lãi) |

**Ví dụ tính toán:**
- Tổng số SELL signals: 44
- Số SELL WIN (Actual_Return < 0): 22
- **Sell Win Rate = 22/44 = 50.0%**

---

## 3. COMBINED WIN RATE (Tỷ lệ thắng tổng hợp)

### Logic:
- Tính trung bình của Buy Win Rate và Sell Win Rate
- Hoặc tính tổng số WIN / tổng số trades

### Công thức:
```
Combined Win Rate = (Buy Win Rate + Sell Win Rate) / 2
```

**Hoặc:**
```
Combined Win Rate = (Số BUY WIN + Số SELL WIN) / (Tổng số BUY + Tổng số SELL)
```

### Ví dụ:
- Buy Win Rate: 73.7%
- Sell Win Rate: 50.0%
- **Combined Win Rate = (73.7% + 50.0%) / 2 = 61.9%**

---

## 4. TẠI SAO SELL WIN KHI Actual_Return < 0?

### Giải thích về SHORT position:

Khi bạn **SELL** (bán khống), bạn:
1. Vay cổ phiếu để bán ngay (giá cao)
2. Chờ giá giảm
3. Mua lại cổ phiếu để trả (giá thấp)
4. **Lãi = Giá bán - Giá mua lại**

**Ví dụ:**
- Ngày 1: SELL signal → Bán khống 100 cổ phiếu NVDA ở giá $500 = $50,000
- Ngày 2: Giá giảm xuống $450 → Mua lại 100 cổ phiếu = $45,000
- **Lãi = $50,000 - $45,000 = $5,000** (tương đương Actual_Return = -10% → WIN)

**Ngược lại:**
- Ngày 1: SELL signal → Bán khống 100 cổ phiếu NVDA ở giá $500 = $50,000
- Ngày 2: Giá tăng lên $550 → Phải mua lại 100 cổ phiếu = $55,000
- **Lỗ = $50,000 - $55,000 = -$5,000** (tương đương Actual_Return = +10% → LOSS)

---

## 5. CODE THỰC TẾ TRONG SCRIPT

```python
# Dòng 162-165: Xác định WIN/LOSS
results_df['Is_Buy_Win'] = (results_df['Signal'] == 2) & (results_df['Actual_Return'] > 0)
results_df['Is_Buy_Loss'] = (results_df['Signal'] == 2) & (results_df['Actual_Return'] <= 0)
results_df['Is_Sell_Win'] = (results_df['Signal'] == 0) & (results_df['Actual_Return'] < 0)
results_df['Is_Sell_Loss'] = (results_df['Signal'] == 0) & (results_df['Actual_Return'] >= 0)

# Dòng 202: Tính Buy Win Rate
buy_wr = len(buy_signals[buy_signals['Actual_Return'] > 0]) / len(buy_signals)

# Dòng 210: Tính Sell Win Rate
sell_wr = len(sell_signals[sell_signals['Actual_Return'] < 0]) / len(sell_signals)

# Dòng 281: Hiển thị kết quả BUY
result = "WIN" if actual > 0 else "LOSS"  # BUY: > 0 = WIN

# Dòng 297: Hiển thị kết quả SELL
result = "WIN" if actual < 0 else "LOSS"  # SELL: < 0 = WIN
```

---

## 6. TÓM TẮT

| Loại Signal | Điều kiện WIN | Điều kiện LOSS | Lý do |
|-------------|---------------|----------------|-------|
| **BUY** | Actual_Return > 0 | Actual_Return <= 0 | Mua → Giá tăng → Lãi |
| **SELL** | Actual_Return < 0 | Actual_Return >= 0 | Bán khống → Giá giảm → Lãi |

**Nhớ:**
- BUY WIN khi giá **TĂNG** (Actual_Return > 0)
- SELL WIN khi giá **GIẢM** (Actual_Return < 0)

