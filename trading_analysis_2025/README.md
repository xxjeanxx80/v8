# Trading Analysis 2025

Folder nay chua script de phan tich ket qua trading tren du lieu nam 2025.

## Chuc nang

- Load model da train (từ artifact)
- Test tren du lieu 2025
- Hien thi chi tiet cac ngay co lenh BUY/SELL
- Export ket qua ra CSV

## Su dung

### Chay phan tich

```bash
cd trading_analysis_2025
python analyze_trading_2025.py
```

### Cac tham so

- `--artifact` hoac `-a`: Duong dan den artifact file (mac dinh: `../nvda_lstm_v7_1_artifact.pth`)
- `--data_dir` hoac `-d`: Duong dan den thu muc data (mac dinh: `../data`)
- `--output` hoac `-o`: Ten file CSV output (mac dinh: `trading_results_2025.csv`)
- `--device`: Device de chay (cpu hoac cuda, mac dinh: cpu)

### Vi du

```bash
# Su dung artifact mac dinh
python analyze_trading_2025.py

# Chi dinh artifact khac
python analyze_trading_2025.py --artifact ../nvda_lstm_v7_artifact.pth

# Su dung GPU
python analyze_trading_2025.py --device cuda
```

## Ket qua

Script se hien thi:

1. **Tong ket ket qua trading**:
   - Tong so ngay
   - So ngay co lenh BUY/SELL/NO_TRADE
   - Win rate cho BUY va SELL
   - Average return va total return

2. **Danh sach chi tiet cac ngay co lenh**:
   - BUY signals: Ngay, Predicted Return, Actual Return, Ket qua (WIN/LOSS)
   - SELL signals: Ngay, Predicted Return, Actual Return, Ket qua (WIN/LOSS)

3. **File CSV**:
   - Luu tat ca ket qua vao file CSV de phan tich sau

## File output

- `trading_results_2025.csv`: File CSV chua ket qua chi tiet tung ngay
- `trading_analysis_2025.png`: Bieu do tong quan ket qua trading (4 subplots)

## Bieu do

Script se tao mot bieu do gom 4 phan:

1. **Cumulative Returns with Buy/Sell Signals**:
   - Duong cumulative returns theo thoi gian
   - Buy signals: tam giac xanh (WIN) hoac do (LOSS)
   - Sell signals: tam giac nguoc xanh duong (WIN) hoac cam (LOSS)

2. **Predicted vs Actual Returns**:
   - Scatter plot so sanh predicted returns (centered) vs actual returns
   - Highlight Buy/Sell signals
   - Duong y=x (perfect prediction)

3. **Signal Distribution**:
   - Bar chart hien thi so luong BUY/SELL/NO_TRADE signals
   - Hien thi phan tram cua moi loai

4. **Win/Loss Analysis**:
   - Stacked bar chart hien thi so luong WIN/LOSS cho BUY va SELL
   - Hien thi win rate cho moi loai signal

## Requirements

De tao bieu do, can cai dat matplotlib:
```bash
pip install matplotlib
```

Neu khong co matplotlib, script van chay binh thuong va chi luu CSV file.

