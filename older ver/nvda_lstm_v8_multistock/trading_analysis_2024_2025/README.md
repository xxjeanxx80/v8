# Trading Analysis 2024-2025

Folder nay chua script de phan tich ket qua trading tren du lieu nam 2024-2025 (V8).

## Chuc nang

- Load model da train (tu artifact V8)
- Test tren du lieu 2024-2025
- Phan tich tong hop va rieng tung nam (2024, 2025)
- Hien thi chi tiet cac ngay co lenh BUY/SELL
- Export ket qua ra CSV (tong hop va rieng tung nam)
- Tao bieu do tong quan cho ca 2 nam

## Su dung

### Chay phan tich

```bash
cd trading_analysis_2024_2025
python analyze_trading_2024_2025.py
```

### Cac tham so

- `--artifact` hoac `-a`: Duong dan den artifact file (mac dinh: `../../nvda_lstm_v8_multistock/nvda_lstm_v8_artifact.pth`)
- `--data_dir` hoac `-d`: Duong dan den thu muc data (mac dinh: `../../data`)
- `--device`: Device de chay (cpu hoac cuda, mac dinh: cpu)
- `--no_plot`: Khong hien thi bieu do (chi luu file PNG)

### Vi du

```bash
# Su dung artifact mac dinh
python analyze_trading_2024_2025.py

# Chi dinh artifact khac
python analyze_trading_2024_2025.py --artifact ../../nvda_lstm_v8_multistock/nvda_lstm_v8_artifact.pth

# Su dung GPU
python analyze_trading_2024_2025.py --device cuda

# Khong hien thi bieu do (chi luu file)
python analyze_trading_2024_2025.py --no_plot
```

## Ket qua

Script se hien thi:

1. **Tong ket ket qua trading (2024-2025)**:
   - Tong so ngay
   - So ngay co lenh BUY/SELL/NO_TRADE
   - Win rate cho BUY va SELL
   - Average return va total return

2. **So sanh ket qua giua 2024 va 2025**:
   - Buy Win Rate, Sell Win Rate, Combined Win Rate cho tung nam
   - Coverage cho tung nam
   - So luong signals cho tung nam

3. **Danh sach chi tiet cac ngay co lenh**:
   - Tong hop 2024-2025
   - Rieng 2024
   - Rieng 2025
   - Moi danh sach bao gom: Ngay, Predicted Return, Actual Return, Ket qua (WIN/LOSS)

4. **File CSV**:
   - `trading_results_2024_2025.csv`: Tong hop ca 2 nam
   - `trading_results_2024.csv`: Rieng nam 2024
   - `trading_results_2025.csv`: Rieng nam 2025

## File output

- `trading_results_2024_2025.csv`: File CSV chua ket qua chi tiet tung ngay (tong hop)
- `trading_results_2024.csv`: File CSV chua ket qua chi tiet tung ngay (rieng 2024)
- `trading_results_2025.csv`: File CSV chua ket qua chi tiet tung ngay (rieng 2025)
- `trading_analysis_2024_2025.png`: Bieu do tong quan ket qua trading (4 subplots cho ca 2 nam)

## Bieu do

Script se tao mot bieu do gom 4 phan:

1. **Cumulative Returns with Buy/Sell Signals (2024-2025)**:
   - Duong cumulative returns theo thoi gian (ca 2 nam)
   - Duong phan cach nam (2025-01-01)
   - Buy signals: tam giac xanh (WIN) hoac do (LOSS)
   - Sell signals: tam giac nguoc xanh duong (WIN) hoac cam (LOSS)

2. **Predicted vs Actual Returns**:
   - Scatter plot so sanh predicted returns (centered) vs actual returns
   - Highlight Buy/Sell signals
   - Duong y=x (perfect prediction)

3. **Signal Distribution (Combined & By Year)**:
   - Grouped bar chart hien thi so luong BUY/SELL/NO_TRADE signals
   - So sanh giua tong hop, 2024, va 2025

4. **Win/Loss Analysis (Combined & By Year)**:
   - Grouped stacked bar chart hien thi so luong WIN/LOSS cho BUY va SELL
   - So sanh giua tong hop, 2024, va 2025
   - Hien thi win rate cho moi loai signal

## Khac biet voi trading_analysis_2025

- Phan tich cho ca 2 nam (2024-2025) thay vi chi 2025
- Phan tich rieng tung nam de so sanh
- Output files rieng cho tung nam
- Bieu do bao gom so sanh giua cac nam
- Su dung V8 artifact va split function

## Requirements

De tao bieu do, can cai dat matplotlib:
```bash
pip install matplotlib
```

Neu khong co matplotlib, script van chay binh thuong va chi luu CSV file.

## Notes

- Script su dung V8 signal generation logic (centered predictions)
- Scalers duoc fit tren pretrain data (nhu trong V8 training) de dam bao consistency
- Metrics duoc verify voi artifact de dam bao accuracy

