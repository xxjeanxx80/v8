#!/usr/bin/env python3
"""
Phan tich ket qua trading tren du lieu 2024-2025
Hien thi chi tiet cac ngay co lenh BUY/SELL
Phan tich tong hop va rieng tung nam
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# Add paths
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
v4_path = os.path.join(workspace_root, 'nvda_lstm_v4_multistock')
v8_path = os.path.join(workspace_root, 'nvda_lstm_v8_multistock')

if v4_path not in sys.path:
    sys.path.insert(0, v4_path)
if v8_path not in sys.path:
    sys.path.insert(0, v8_path)

try:
    from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete, LSTMRegressor  # type: ignore[import]
except Exception:
    import importlib.util
    spec_path = os.path.join(v4_path, 'nvda_lstm_multistock_complete.py')
    if os.path.exists(spec_path):
        spec = importlib.util.spec_from_file_location('nvda_lstm_multistock_complete', spec_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        NVDA_MultiStock_Complete = getattr(mod, 'NVDA_MultiStock_Complete')
        LSTMRegressor = getattr(mod, 'LSTMRegressor')
    else:
        raise ImportError(f'Module file not found: {spec_path}')

# Import v8 split function
try:
    from nvda_lstm_v8_multistock import split_data_by_years_v8
except Exception:
    import importlib.util
    v8_script_path = os.path.join(v8_path, 'nvda_lstm_v8_multistock.py')
    if os.path.exists(v8_script_path):
        spec = importlib.util.spec_from_file_location('nvda_lstm_v8_multistock', v8_script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        split_data_by_years_v8 = getattr(mod, 'split_data_by_years_v8')
    else:
        raise ImportError(f'Module file not found: {v8_script_path}')

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def load_artifact(artifact_path):
    """Load artifact file"""
    if not os.path.exists(artifact_path):
        return None
    return torch.load(artifact_path, map_location='cpu', weights_only=False)


def load_test_data_2024_2025(data_dir, predictor, feature_cols, sequence_length=30):
    """Load va prepare test data cho 2024-2025"""
    # Load multi-stock data
    df, _ = predictor.load_multi_stock_data(data_dir)
    
    # Filter chi NVDA va 2024-2025
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Normalize timezone
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    
    # Filter NVDA only
    nvda_mask = df['stock_NVDA'] == 1
    df_nvda = df[nvda_mask].copy()
    
    # Filter 2024-2025
    test_start = pd.to_datetime('2024-01-01')
    test_end = pd.to_datetime('2025-12-31')
    df_test = df_nvda[(df_nvda['Date'] >= test_start) & (df_nvda['Date'] <= test_end)].copy()
    df_test = df_test.sort_values('Date').reset_index(drop=True)
    
    if len(df_test) == 0:
        raise ValueError("Khong co du lieu 2024-2025 trong dataset")
    
    # Prepare features and targets
    X_test = df_test[feature_cols].values
    y_test_reg = df_test['future_return'].values.reshape(-1, 1)
    dates = df_test['Date'].values
    
    # Create sequences
    X_test_seq = []
    y_test_seq = []
    dates_seq = []
    
    for i in range(sequence_length, len(X_test)):
        X_test_seq.append(X_test[i - sequence_length:i])
        y_test_seq.append(y_test_reg[i])
        dates_seq.append(dates[i])
    
    X_test_seq = np.array(X_test_seq)
    y_test_seq = np.array(y_test_seq)
    dates_seq = np.array(dates_seq)
    
    return X_test_seq, y_test_seq, dates_seq, df_test


def predict_and_analyze(model, X_test, y_test, dates, scaler_X, scaler_y, 
                        buy_thr, sell_thr, device='cpu'):
    """
    Predict va phan tich ket qua
    
    CRITICAL: V8 signal generation logic (must match training/testing):
    1. Predict future returns → y_pred (raw predictions)
    2. Center predictions: y_pred_centered = y_pred - mean(y_pred)
    3. Generate signals using CENTERED predictions:
       - BUY  if y_pred_centered > buy_threshold
       - SELL if y_pred_centered < sell_threshold
       - NO_TRADE otherwise
    
    This ensures consistency with v8 artifact results.
    """
    # Predict
    model.eval()
    with torch.no_grad():
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        preds_scaled = model(X_test_tensor).cpu().numpy()
    
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true = y_test.flatten()
    
    # V8 STEP 2: Center predictions (CRITICAL for threshold matching)
    y_pred_mean = np.mean(preds)
    y_pred_centered = preds - y_pred_mean
    
    # V8 STEP 3: Generate signals using CENTERED predictions (not raw!)
    signals = np.where(y_pred_centered > buy_thr, 2,
                     np.where(y_pred_centered < sell_thr, 0, 1))
    
    # Create detailed dataframe
    results_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Return': preds,
        'Predicted_Return_Centered': y_pred_centered,
        'Actual_Return': y_true,
        'Signal': signals,
        'Buy_Threshold': buy_thr,
        'Sell_Threshold': sell_thr
    })
    
    # Add signal labels
    results_df['Signal_Label'] = results_df['Signal'].map({0: 'SELL', 1: 'NO_TRADE', 2: 'BUY'})
    
    # Add return analysis
    results_df['Is_Buy_Win'] = (results_df['Signal'] == 2) & (results_df['Actual_Return'] > 0)
    results_df['Is_Buy_Loss'] = (results_df['Signal'] == 2) & (results_df['Actual_Return'] <= 0)
    results_df['Is_Sell_Win'] = (results_df['Signal'] == 0) & (results_df['Actual_Return'] < 0)
    results_df['Is_Sell_Loss'] = (results_df['Signal'] == 0) & (results_df['Actual_Return'] >= 0)
    
    # Add year column for analysis
    results_df['Year'] = pd.to_datetime(results_df['Date']).dt.year
    
    return results_df


def analyze_by_year(results_df):
    """Phan tich rieng cho 2024 va 2025"""
    results_2024 = results_df[results_df['Year'] == 2024].copy()
    results_2025 = results_df[results_df['Year'] == 2025].copy()
    
    return {
        '2024': results_2024,
        '2025': results_2025,
        'combined': results_df
    }


def print_trading_summary(results_df, year_label="2024-2025"):
    """In tong ket ket qua trading"""
    print(f"\n{'='*80}")
    print(f"TONG KET KET QUA TRADING NAM {year_label}")
    print(f"{'='*80}")
    
    total_days = len(results_df)
    buy_signals = results_df[results_df['Signal'] == 2]
    sell_signals = results_df[results_df['Signal'] == 0]
    no_trade = results_df[results_df['Signal'] == 1]
    
    print(f"\nTong so ngay: {total_days}")
    print(f"  - BUY signals: {len(buy_signals)} ({len(buy_signals)/total_days:.1%})")
    print(f"  - SELL signals: {len(sell_signals)} ({len(sell_signals)/total_days:.1%})")
    print(f"  - NO_TRADE: {len(no_trade)} ({len(no_trade)/total_days:.1%})")
    
    if len(buy_signals) > 0:
        buy_wr = len(buy_signals[buy_signals['Actual_Return'] > 0]) / len(buy_signals)
        avg_buy_return = buy_signals['Actual_Return'].mean()
        print(f"\nBUY Signals:")
        print(f"  - Win Rate: {buy_wr:.1%}")
        print(f"  - Average Return: {avg_buy_return:.4f} ({avg_buy_return*100:.2f}%)")
        print(f"  - Total Return: {buy_signals['Actual_Return'].sum():.4f} ({buy_signals['Actual_Return'].sum()*100:.2f}%)")
    
    if len(sell_signals) > 0:
        sell_wr = len(sell_signals[sell_signals['Actual_Return'] < 0]) / len(sell_signals)
        avg_sell_return = sell_signals['Actual_Return'].mean()
        print(f"\nSELL Signals:")
        print(f"  - Win Rate: {sell_wr:.1%}")
        print(f"  - Average Return: {avg_sell_return:.4f} ({avg_sell_return*100:.2f}%)")
        print(f"  - Total Return: {sell_signals['Actual_Return'].sum():.4f} ({sell_signals['Actual_Return'].sum()*100:.2f}%)")
    
    # Combined metrics
    all_trades = results_df[results_df['Signal'] != 1]
    if len(all_trades) > 0:
        combined_wr = (len(buy_signals[buy_signals['Actual_Return'] > 0]) + 
                      len(sell_signals[sell_signals['Actual_Return'] < 0])) / len(all_trades)
        print(f"\nCombined Metrics:")
        print(f"  - Combined Win Rate: {combined_wr:.1%}")
        print(f"  - Total Trading Return: {all_trades['Actual_Return'].sum():.4f} ({all_trades['Actual_Return'].sum()*100:.2f}%)")


def print_year_comparison(results_by_year):
    """So sanh ket qua giua 2024 va 2025"""
    print(f"\n{'='*80}")
    print("SO SANH KET QUA GIUA 2024 VA 2025")
    print(f"{'='*80}")
    
    for year in ['2024', '2025']:
        results = results_by_year[year]
        if len(results) == 0:
            print(f"\n{year}: Khong co du lieu")
            continue
        
        buy_signals = results[results['Signal'] == 2]
        sell_signals = results[results['Signal'] == 0]
        
        buy_wr = len(buy_signals[buy_signals['Actual_Return'] > 0]) / len(buy_signals) if len(buy_signals) > 0 else 0
        sell_wr = len(sell_signals[sell_signals['Actual_Return'] < 0]) / len(sell_signals) if len(sell_signals) > 0 else 0
        combined_wr = (buy_wr + sell_wr) / 2 if (len(buy_signals) > 0 or len(sell_signals) > 0) else 0
        coverage = (len(buy_signals) + len(sell_signals)) / len(results)
        
        print(f"\n{year}:")
        print(f"  - Buy Win Rate: {buy_wr:.1%}")
        print(f"  - Sell Win Rate: {sell_wr:.1%}")
        print(f"  - Combined Win Rate: {combined_wr:.1%}")
        print(f"  - Coverage: {coverage:.1%}")
        print(f"  - Total Days: {len(results)}")
        print(f"  - BUY Signals: {len(buy_signals)}")
        print(f"  - SELL Signals: {len(sell_signals)}")


def print_trading_dates(results_df, year_label=""):
    """In danh sach cac ngay co lenh BUY/SELL"""
    if year_label:
        print(f"\n{'='*80}")
        print(f"DANH SACH CAC NGAY CO LENH TRADING {year_label}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("DANH SACH CAC NGAY CO LENH TRADING")
        print(f"{'='*80}")
    
    # BUY signals
    buy_signals = results_df[results_df['Signal'] == 2].copy()
    if len(buy_signals) > 0:
        print(f"\n{'='*80}")
        print(f"BUY SIGNALS ({len(buy_signals)} ngay):")
        print(f"{'='*80}")
        print(f"{'Ngay':<12} {'Predicted':<12} {'Actual':<12} {'Ket Qua':<10}")
        print("-" * 80)
        
        for idx, row in buy_signals.iterrows():
            date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
            pred = row['Predicted_Return']
            actual = row['Actual_Return']
            result = "WIN" if actual > 0 else "LOSS"
            print(f"{date_str:<12} {pred:>11.4f} {actual:>11.4f} {result:<10}")
    
    # SELL signals
    sell_signals = results_df[results_df['Signal'] == 0].copy()
    if len(sell_signals) > 0:
        print(f"\n{'='*80}")
        print(f"SELL SIGNALS ({len(sell_signals)} ngay):")
        print(f"{'='*80}")
        print(f"{'Ngay':<12} {'Predicted':<12} {'Actual':<12} {'Ket Qua':<10}")
        print("-" * 80)
        
        for idx, row in sell_signals.iterrows():
            date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
            pred = row['Predicted_Return']
            actual = row['Actual_Return']
            result = "WIN" if actual < 0 else "LOSS"
            print(f"{date_str:<12} {pred:>11.4f} {actual:>11.4f} {result:<10}")


def create_visualizations(results_df, results_by_year, output_dir=None):
    """
    Tao cac bieu do de xem tong quan ket qua trading cho 2024-2025
    
    Cac bieu do:
    1. Cumulative returns theo thoi gian voi Buy/Sell signals (ca 2 nam)
    2. Predicted vs Actual returns
    3. Signal distribution (tong hop va rieng tung nam)
    4. Win/Loss analysis (tong hop va rieng tung nam)
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    # Chuan bi du lieu
    results_df = results_df.copy()
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    results_df = results_df.sort_values('Date').reset_index(drop=True)
    
    # Tinh cumulative returns
    results_df['Cumulative_Return'] = results_df['Actual_Return'].cumsum()
    
    # Tach Buy/Sell signals
    buy_signals = results_df[results_df['Signal'] == 2]
    sell_signals = results_df[results_df['Signal'] == 0]
    
    # Tao figure voi 4 subplots
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('NVDA LSTM v8 - Trading Analysis 2024-2025', fontsize=16, fontweight='bold')
    
    # ========== Subplot 1: Cumulative Returns voi Buy/Sell Signals (ca 2 nam) ==========
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(results_df['Date'], results_df['Cumulative_Return'], 
             linewidth=2, label='Cumulative Return', color='black')
    
    # Ve duong phan cach nam
    year_boundary = pd.to_datetime('2025-01-01')
    if year_boundary in results_df['Date'].values:
        ax1.axvline(x=year_boundary, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='2025 Start')
    
    # Ve Buy signals
    if len(buy_signals) > 0:
        buy_wins = buy_signals[buy_signals['Actual_Return'] > 0]
        buy_losses = buy_signals[buy_signals['Actual_Return'] <= 0]
        
        if len(buy_wins) > 0:
            ax1.scatter(buy_wins['Date'], buy_wins['Cumulative_Return'], 
                       color='green', marker='^', s=100, label=f'BUY WIN ({len(buy_wins)})', 
                       zorder=5, edgecolors='darkgreen', linewidths=1.5)
        if len(buy_losses) > 0:
            ax1.scatter(buy_losses['Date'], buy_losses['Cumulative_Return'], 
                       color='red', marker='^', s=100, label=f'BUY LOSS ({len(buy_losses)})', 
                       zorder=5, edgecolors='darkred', linewidths=1.5)
    
    # Ve Sell signals
    if len(sell_signals) > 0:
        sell_wins = sell_signals[sell_signals['Actual_Return'] < 0]
        sell_losses = sell_signals[sell_signals['Actual_Return'] >= 0]
        
        if len(sell_wins) > 0:
            ax1.scatter(sell_wins['Date'], sell_wins['Cumulative_Return'], 
                       color='blue', marker='v', s=100, label=f'SELL WIN ({len(sell_wins)})', 
                       zorder=5, edgecolors='darkblue', linewidths=1.5)
        if len(sell_losses) > 0:
            ax1.scatter(sell_losses['Date'], sell_losses['Cumulative_Return'], 
                       color='orange', marker='v', s=100, label=f'SELL LOSS ({len(sell_losses)})', 
                       zorder=5, edgecolors='darkorange', linewidths=1.5)
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Cumulative Return', fontsize=10)
    ax1.set_title('Cumulative Returns with Buy/Sell Signals (2024-2025)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========== Subplot 2: Predicted vs Actual Returns ==========
    ax2 = plt.subplot(2, 2, 2)
    
    # Scatter plot: Predicted vs Actual
    ax2.scatter(results_df['Predicted_Return_Centered'], results_df['Actual_Return'], 
               alpha=0.5, s=30, color='blue', label='All Predictions')
    
    # Highlight Buy signals
    if len(buy_signals) > 0:
        ax2.scatter(buy_signals['Predicted_Return_Centered'], buy_signals['Actual_Return'], 
                   color='green', marker='^', s=80, label='BUY Signals', 
                   edgecolors='darkgreen', linewidths=1.5, zorder=5)
    
    # Highlight Sell signals
    if len(sell_signals) > 0:
        ax2.scatter(sell_signals['Predicted_Return_Centered'], sell_signals['Actual_Return'], 
                   color='red', marker='v', s=80, label='SELL Signals', 
                   edgecolors='darkred', linewidths=1.5, zorder=5)
    
    # Ve duong y=x (perfect prediction)
    min_val = min(results_df['Predicted_Return_Centered'].min(), results_df['Actual_Return'].min())
    max_val = max(results_df['Predicted_Return_Centered'].max(), results_df['Actual_Return'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    ax2.set_xlabel('Predicted Return (Centered)', fontsize=10)
    ax2.set_ylabel('Actual Return', fontsize=10)
    ax2.set_title('Predicted vs Actual Returns', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ========== Subplot 3: Signal Distribution (tong hop va rieng tung nam) ==========
    ax3 = plt.subplot(2, 2, 3)
    
    # Tinh toan cho tong hop va rieng tung nam
    signal_counts_combined = results_df['Signal_Label'].value_counts()
    signal_counts_2024 = results_by_year['2024']['Signal_Label'].value_counts() if len(results_by_year['2024']) > 0 else pd.Series()
    signal_counts_2025 = results_by_year['2025']['Signal_Label'].value_counts() if len(results_by_year['2025']) > 0 else pd.Series()
    
    # Grouped bar chart
    categories = ['BUY', 'SELL', 'NO_TRADE']
    colors = {'BUY': 'green', 'SELL': 'red', 'NO_TRADE': 'gray'}
    
    x = np.arange(len(categories))
    width = 0.25
    
    combined_vals = [signal_counts_combined.get(cat, 0) for cat in categories]
    vals_2024 = [signal_counts_2024.get(cat, 0) for cat in categories]
    vals_2025 = [signal_counts_2025.get(cat, 0) for cat in categories]
    
    ax3.bar(x - width, combined_vals, width, label='Combined', color='blue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.bar(x, vals_2024, width, label='2024', color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.bar(x + width, vals_2025, width, label='2025', color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax3.set_xlabel('Signal Type', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Signal Distribution (Combined & By Year)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== Subplot 4: Win/Loss Analysis (tong hop va rieng tung nam) ==========
    ax4 = plt.subplot(2, 2, 4)
    
    # Tinh toan metrics cho tong hop
    buy_wins_combined = len(buy_signals[buy_signals['Actual_Return'] > 0]) if len(buy_signals) > 0 else 0
    buy_losses_combined = len(buy_signals[buy_signals['Actual_Return'] <= 0]) if len(buy_signals) > 0 else 0
    sell_wins_combined = len(sell_signals[sell_signals['Actual_Return'] < 0]) if len(sell_signals) > 0 else 0
    sell_losses_combined = len(sell_signals[sell_signals['Actual_Return'] >= 0]) if len(sell_signals) > 0 else 0
    
    # Tinh toan cho 2024
    results_2024 = results_by_year['2024']
    buy_signals_2024 = results_2024[results_2024['Signal'] == 2] if len(results_2024) > 0 else pd.DataFrame()
    sell_signals_2024 = results_2024[results_2024['Signal'] == 0] if len(results_2024) > 0 else pd.DataFrame()
    buy_wins_2024 = len(buy_signals_2024[buy_signals_2024['Actual_Return'] > 0]) if len(buy_signals_2024) > 0 else 0
    buy_losses_2024 = len(buy_signals_2024[buy_signals_2024['Actual_Return'] <= 0]) if len(buy_signals_2024) > 0 else 0
    sell_wins_2024 = len(sell_signals_2024[sell_signals_2024['Actual_Return'] < 0]) if len(sell_signals_2024) > 0 else 0
    sell_losses_2024 = len(sell_signals_2024[sell_signals_2024['Actual_Return'] >= 0]) if len(sell_signals_2024) > 0 else 0
    
    # Tinh toan cho 2025
    results_2025 = results_by_year['2025']
    buy_signals_2025 = results_2025[results_2025['Signal'] == 2] if len(results_2025) > 0 else pd.DataFrame()
    sell_signals_2025 = results_2025[results_2025['Signal'] == 0] if len(results_2025) > 0 else pd.DataFrame()
    buy_wins_2025 = len(buy_signals_2025[buy_signals_2025['Actual_Return'] > 0]) if len(buy_signals_2025) > 0 else 0
    buy_losses_2025 = len(buy_signals_2025[buy_signals_2025['Actual_Return'] <= 0]) if len(buy_signals_2025) > 0 else 0
    sell_wins_2025 = len(sell_signals_2025[sell_signals_2025['Actual_Return'] < 0]) if len(sell_signals_2025) > 0 else 0
    sell_losses_2025 = len(sell_signals_2025[sell_signals_2025['Actual_Return'] >= 0]) if len(sell_signals_2025) > 0 else 0
    
    # Grouped stacked bar chart
    categories = ['Combined\nBUY', 'Combined\nSELL', '2024\nBUY', '2024\nSELL', '2025\nBUY', '2025\nSELL']
    win_counts = [buy_wins_combined, sell_wins_combined, buy_wins_2024, sell_wins_2024, buy_wins_2025, sell_wins_2025]
    loss_counts = [buy_losses_combined, sell_losses_combined, buy_losses_2024, sell_losses_2024, buy_losses_2025, sell_losses_2025]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars1 = ax4.bar(x, win_counts, width, label='WIN', color='green', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x, loss_counts, width, bottom=win_counts, label='LOSS', color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Them so luong len tren moi bar
    for i, (w, l) in enumerate(zip(win_counts, loss_counts)):
        total = w + l
        if total > 0:
            win_pct = w / total * 100
            if w > 0:
                ax4.text(i, w/2, f'{w}\n({win_pct:.0f}%)', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
            if l > 0:
                ax4.text(i, w + l/2, f'{l}\n({100-win_pct:.0f}%)', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
    
    ax4.set_xlabel('Category', fontsize=10)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Win/Loss Analysis (Combined & By Year)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=8)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Luu bieu do
    output_path = os.path.join(output_dir, 'trading_analysis_2024_2025.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Bieu do da duoc luu vao: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Phan tich ket qua trading tren du lieu 2024-2025')
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    default_artifact = os.path.join(workspace_root, 'nvda_lstm_v8_multistock', 'nvda_lstm_v8_artifact.pth')
    default_data_dir = os.path.join(workspace_root, 'data')
    
    parser.add_argument('--artifact', '-a', default=default_artifact,
                       help='Path to artifact file (default: nvda_lstm_v8_artifact.pth)')
    parser.add_argument('--data_dir', '-d', default=default_data_dir,
                       help='Path to data directory')
    parser.add_argument('--device', default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Khong hien thi bieu do (chi luu file)')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA khong co san, su dung CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"\n{'='*80}")
    print("PHAN TICH KET QUA TRADING NAM 2024-2025")
    print(f"{'='*80}")
    
    # Load artifact - tim o nhieu vi tri neu can
    print(f"\nDang load artifact: {args.artifact}")
    artifact = load_artifact(args.artifact)
    
    # Neu khong tim thay, thu tim o cac vi tri khac
    if artifact is None:
        print(f"  Khong tim thay artifact tai: {args.artifact}")
        print(f"  Dang tim o cac vi tri khac...")
        
        # Tim o cac vi tri co the (chi V7/V8 vi co validation_thresholds)
        possible_paths = [
            os.path.join(workspace_root, 'nvda_lstm_v8_multistock', 'nvda_lstm_v8_artifact.pth'),
            os.path.join(workspace_root, 'nvda_lstm_v7_multistock', 'nvda_lstm_v7_1_artifact.pth'),
            os.path.join(workspace_root, 'nvda_lstm_v7_multistock', 'nvda_lstm_v7_multistock_artifact.pth'),
        ]
        
        artifact_found = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  Tim thay artifact tai: {path}")
                artifact = load_artifact(path)
                if artifact is not None:
                    artifact_found = True
                    print(f"  OK: Da load artifact tu: {path}")
                    break
        
        if not artifact_found:
            print(f"\n{'='*80}")
            print("LOI: Khong tim thay artifact nao!")
            print(f"{'='*80}")
            print(f"\nVui long chay script V8 truoc de tao artifact:")
            print(f"  cd {os.path.join(workspace_root, 'nvda_lstm_v8_multistock')}")
            print(f"  python nvda_lstm_v8_multistock.py")
            print(f"\nHoac chi dinh duong dan den artifact file:")
            print(f"  python analyze_trading_2024_2025.py --artifact <duong_dan_den_artifact.pth>")
            raise FileNotFoundError(f"Khong tim thay artifact. Vui long chay V8 script truoc hoac chi dinh artifact path.")
    else:
        print("  OK: Artifact loaded")
    
    # Extract information from artifact
    features = artifact.get('features', [])
    validation_thresholds = artifact.get('validation_thresholds', {})
    test_metrics = artifact.get('test_metrics', {})
    test_by_year = artifact.get('test_by_year', {})
    
    # Kiem tra xem artifact co validation_thresholds khong (chi V7/V8 co)
    if not validation_thresholds:
        print(f"\n{'='*80}")
        print("LOI: Artifact khong co validation_thresholds!")
        print(f"{'='*80}")
        print(f"\nScript nay chi ho tro V7/V8 artifacts (co validation_thresholds).")
        print(f"Artifact V5/V6 khong co cau truc nay.")
        print(f"\nVui long chay script V8 truoc de tao artifact:")
        print(f"  cd {os.path.join(workspace_root, 'nvda_lstm_v8_multistock')}")
        print(f"  python nvda_lstm_v8_multistock.py")
        print(f"\nHoac su dung V7.1 artifact:")
        print(f"  python analyze_trading_2024_2025.py --artifact <duong_dan_den_v7_1_artifact.pth>")
        raise ValueError("Artifact khong co validation_thresholds. Chi V7/V8 artifacts duoc ho tro.")
    
    buy_thr = validation_thresholds.get('buy_thr')
    sell_thr = validation_thresholds.get('sell_thr')
    
    if buy_thr is None or sell_thr is None:
        print(f"\n{'='*80}")
        print("LOI: Artifact co validation_thresholds nhung thieu buy_thr hoac sell_thr!")
        print(f"{'='*80}")
        raise ValueError("Artifact validation_thresholds khong day du (thieu buy_thr hoac sell_thr)")
    
    print(f"  Features: {len(features)}")
    print(f"  Buy Threshold: {buy_thr:.4f}")
    print(f"  Sell Threshold: {sell_thr:.4f}")
    
    # Load model
    print(f"\nDang load model...")
    sequence_length = 30
    model = LSTMRegressor(input_size=len(features)).to(device)
    model.load_state_dict(artifact['model_state'])
    model.eval()
    print("  OK: Model loaded")
    
    # CRITICAL: Recreate scalers from PRETRAIN data (same as v8 training)
    print(f"\nDang load du lieu de tao scalers (pretrain data)...")
    predictor = NVDA_MultiStock_Complete(sequence_length=sequence_length, horizon=5)
    
    # Load full dataset to recreate data splits (same as v8)
    df_full, _ = predictor.load_multi_stock_data(args.data_dir)
    
    # Normalize timezone
    if not pd.api.types.is_datetime64_any_dtype(df_full['Date']):
        df_full['Date'] = pd.to_datetime(df_full['Date'])
    df_full['Date'] = pd.to_datetime(df_full['Date']).dt.tz_localize(None)
    
    # Recreate data splits (same logic as v8)
    actual_min_date = df_full['Date'].min()
    target_date_2015 = pd.to_datetime('2015-01-01')
    if actual_min_date > target_date_2015:
        pretrain_start = actual_min_date.strftime('%Y-01-01')
        pretrain_end_candidate = actual_min_date + pd.DateOffset(years=2)
        if pretrain_end_candidate.tz is not None:
            pretrain_end_candidate = pretrain_end_candidate.tz_localize(None)
        pretrain_end_date = min(pd.to_datetime('2020-12-31'), pretrain_end_candidate)
        pretrain_end = pretrain_end_date.strftime('%Y-12-31')
    else:
        pretrain_start = '2015-01-01'
        pretrain_end = '2020-12-31'
    
    # Split data (same as v8)
    data_splits = split_data_by_years_v8(
        df_full, features, predictor,
        pretrain_start=pretrain_start,
        pretrain_end=pretrain_end,
        finetune_start='2021-01-01',
        finetune_end='2023-12-31',
        val_start='2022-01-01',
        val_end='2023-12-31',
        test_start='2024-01-01',
        test_end='2025-12-31'
    )
    
    # Fit scalers on PRETRAIN data (CRITICAL: same as v8 training)
    X_pretrain, y_pretrain_reg, _ = data_splits['pretrain']
    if X_pretrain is None or len(X_pretrain) == 0:
        # Fallback: use finetune data if pretrain is empty
        X_pretrain, y_pretrain_reg, _ = data_splits['finetune']
        if X_pretrain is None or len(X_pretrain) == 0:
            raise RuntimeError("Cannot recreate scalers: no pretrain or finetune data available")
    
    # Split pretrain: 80% train (for scaler fitting), 20% val
    pretrain_split_idx = int(len(X_pretrain) * 0.8)
    X_pretrain_train = X_pretrain[:pretrain_split_idx]
    y_pretrain_train = y_pretrain_reg[:pretrain_split_idx]
    
    # Create scalers and fit on pretrain TRAIN data (same as v8)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_pretrain_flat = X_pretrain_train.reshape(-1, X_pretrain_train.shape[-1])
    scaler_X.fit(X_pretrain_flat)
    
    y_pretrain_flat = y_pretrain_train.flatten()
    scaler_y.fit(y_pretrain_flat.reshape(-1, 1))
    
    print(f"  OK: Scalers fitted on pretrain data ({len(X_pretrain_train)} sequences)")
    
    # Now load test data
    print(f"\nDang load du lieu test 2024-2025...")
    X_test, y_test, dates, df_test = load_test_data_2024_2025(
        args.data_dir, predictor, features, sequence_length
    )
    print(f"  OK: Loaded {len(X_test)} test sequences")
    print(f"  Date range: {pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()}")
    
    # Predict and analyze
    print(f"\nDang predict va phan tich...")
    results_df = predict_and_analyze(
        model, X_test, y_test, dates, scaler_X, scaler_y,
        buy_thr, sell_thr, device
    )
    print("  OK: Analysis completed")
    
    # Phan tich rieng tung nam
    results_by_year = analyze_by_year(results_df)
    
    # Verify consistency with artifact metrics
    print(f"\n{'='*80}")
    print("VERIFICATION: So sanh voi artifact metrics")
    print(f"{'='*80}")
    
    buy_signals = results_df[results_df['Signal'] == 2]
    sell_signals = results_df[results_df['Signal'] == 0]
    
    buy_wr = len(buy_signals[buy_signals['Actual_Return'] > 0]) / len(buy_signals) if len(buy_signals) > 0 else 0
    sell_wr = len(sell_signals[sell_signals['Actual_Return'] < 0]) / len(sell_signals) if len(sell_signals) > 0 else 0
    combined_wr = (buy_wr + sell_wr) / 2 if (len(buy_signals) > 0 or len(sell_signals) > 0) else 0
    coverage = (len(buy_signals) + len(sell_signals)) / len(results_df)
    
    print(f"\nAnalysis Results (Combined 2024-2025):")
    print(f"  Buy Win Rate: {buy_wr:.1%}")
    print(f"  Sell Win Rate: {sell_wr:.1%}")
    print(f"  Combined Win Rate: {combined_wr:.1%}")
    print(f"  Coverage: {coverage:.1%}")
    
    if test_metrics:
        print(f"\nArtifact Test Metrics (from v8):")
        print(f"  Buy Win Rate: {test_metrics.get('buy_wr', 0):.1%}")
        print(f"  Sell Win Rate: {test_metrics.get('sell_wr', 0):.1%}")
        print(f"  Combined Win Rate: {test_metrics.get('combined_wr', 0):.1%}")
        print(f"  Coverage: {test_metrics.get('coverage', 0):.1%}")
        
        # Check if metrics match
        wr_diff = abs(buy_wr - test_metrics.get('buy_wr', 0))
        if wr_diff < 0.01:
            print(f"\n  OK: Metrics match artifact (difference < 1%)")
        else:
            print(f"\n  WARNING: Metrics differ from artifact (difference: {wr_diff:.1%})")
    
    # Print summary tong hop
    print_trading_summary(results_df, "2024-2025")
    
    # Print so sanh giua 2024 va 2025
    print_year_comparison(results_by_year)
    
    # Print detailed dates cho tong hop
    print_trading_dates(results_df)
    
    # Print detailed dates cho rieng tung nam
    if len(results_by_year['2024']) > 0:
        print_trading_dates(results_by_year['2024'], "(2024)")
    if len(results_by_year['2025']) > 0:
        print_trading_dates(results_by_year['2025'], "(2025)")
    
    # Save to CSV files
    output_dir = os.path.dirname(__file__)
    
    # Tong hop
    output_path_combined = os.path.join(output_dir, 'trading_results_2024_2025.csv')
    results_df.to_csv(output_path_combined, index=False)
    print(f"\n{'='*80}")
    print(f"Ket qua tong hop da duoc luu vao: {output_path_combined}")
    
    # Rieng 2024
    if len(results_by_year['2024']) > 0:
        output_path_2024 = os.path.join(output_dir, 'trading_results_2024.csv')
        results_by_year['2024'].to_csv(output_path_2024, index=False)
        print(f"Ket qua 2024 da duoc luu vao: {output_path_2024}")
    
    # Rieng 2025
    if len(results_by_year['2025']) > 0:
        output_path_2025 = os.path.join(output_dir, 'trading_results_2025.csv')
        results_by_year['2025'].to_csv(output_path_2025, index=False)
        print(f"Ket qua 2025 da duoc luu vao: {output_path_2025}")
    
    print(f"{'='*80}")
    
    # Tao bieu do
    print(f"\n{'='*80}")
    print("Dang tao bieu do...")
    print(f"{'='*80}")
    try:
        fig = create_visualizations(results_df, results_by_year, output_dir=output_dir)
        print("  OK: Bieu do da duoc tao thanh cong")
        
        # Hien thi bieu do neu khong co flag --no_plot
        if not args.no_plot:
            try:
                plt.show()
            except Exception:
                print("  Note: Khong the hien thi bieu do (co the do khong co display)")
                print("  Bieu do da duoc luu vao file, co the mo bang image viewer")
        else:
            plt.close(fig)
            print("  Note: Bieu do da duoc luu, khong hien thi (--no_plot flag)")
    except ImportError:
        print("  WARNING: matplotlib khong duoc cai dat")
        print("  Cai dat: pip install matplotlib")
        print("  CSV file van co the su dung.")
    except Exception as e:
        print(f"  WARNING: Khong the tao bieu do: {e}")
        print("  CSV file van co the su dung.")


if __name__ == '__main__':
    main()

