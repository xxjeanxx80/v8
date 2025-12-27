#!/usr/bin/env python3
"""
NVDA LSTM v7.2 - No-Leakage Strategy with Fixed Logic

V7.2 khac voi V7.1:
1. Loai bo cac feature co gia tri tuyet doi: bb_lower, bb_middle, bb_upper, obv (tranh lam hong Scaler)
2. Them Stoploss gia lap: STOP_LOSS_PCT = -7%, force exit khi daily_return < -7%
3. Sua tinh Drawdown: Tinh dua tren Equity Curve (Von) thay vi cumulative_returns
4. Siet chat thuat toan toi uu: Chi chap nhan bo tham so neu ca buy_expectancy VA sell_expectancy > 0

Muc dich:
- Giam overfit: Tranh data leakage (V7 co the leak vi pretrain 2015-2025 bao gom test period 2025)
- Tang generalization: Pretrain tren period khac hoan toan voi test period
- Out-of-sample thuc su: Test period 2025 khong xuat hien trong pretrain/finetune
- Sua loi logic nghiem trong: Feature filtering, Drawdown calculation, Optimization constraints

Cai tien tu V7.1:
- V7.1: Expectancy proxy scoring (thay vi chi win rate)
- V7.2: Center y_pred truoc threshold (threshold on dinh hon)
- V7.3: Asymmetric scoring cho NVDA (uu tien BUY: 70/30)
- V7.2 Fix: Loai bo absolute features, stoploss simulation, drawdown fix, strict optimization
- Overfitting detection: Walk-forward analysis
- False signal filtering: Loai bo signals co win rate thap, return am, expectancy < 0
- Trade/no-trade logic: Hybrid (confidence threshold + min expectancy)
- Profitability verification: Kiem tra lai/lo thuc te
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# allow importing v4 utilities (robustly)
v4_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nvda_lstm_v4_multistock'))
if v4_path not in sys.path:
    sys.path.insert(0, v4_path)

try:
    from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete, LSTMRegressor, RegressionDataset  # type: ignore[import]
except Exception:
    import importlib.util
    spec_path = os.path.join(v4_path, 'nvda_lstm_multistock_complete.py')
    if os.path.exists(spec_path):
        spec = importlib.util.spec_from_file_location('nvda_lstm_multistock_complete', spec_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Unable to load specification for {spec_path}')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        NVDA_MultiStock_Complete = getattr(mod, 'NVDA_MultiStock_Complete')
        LSTMRegressor = getattr(mod, 'LSTMRegressor')
        RegressionDataset = getattr(mod, 'RegressionDataset')
    else:
        raise ImportError(f'Module file not found: {spec_path}')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==================== V7 Configuration ====================
# Asymmetric scoring cho NVDA (uu tien BUY)
BUY_WEIGHT = 0.7
SELL_WEIGHT = 0.3

# False signal thresholds
MIN_WIN_RATE = 0.50
MIN_EXPECTANCY = 0.01

# Trade/no-trade logic
CONFIDENCE_THRESHOLD_PCT = 60  # Top 40% confidence

# Overfitting check
WALK_FORWARD_WINDOWS = 3

# V7.2: Stoploss gia lap
STOP_LOSS_PCT = -0.07  # -7% stoploss


def create_sequences(X, y, sequence_length):
    """Tao sequences cho LSTM"""
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i - sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def split_data_by_years_v7_1(df, feature_cols, predictor,
                             pretrain_start='2015-01-01', pretrain_end='2020-12-31',
                             finetune_start='2021-01-01', finetune_end='2024-12-31',
                             val_start='2023-01-01', val_end='2024-12-31',
                             test_start='2025-01-01', test_end='2025-12-31'):
    """
    V7.1: Split data KHONG co leakage
    - Pretrain: 2015-2020 (5 nam) - all stocks, TACH BIET voi test
    - Fine-tune: 2021-2024 (4 nam) - all stocks, TACH BIET voi test
    - Validation: 2023-2024 (2 nam) - NVDA only (cho threshold optimization)
    - Test: 2025 (1 nam) - NVDA only, OUT-OF-SAMPLE thuc su
    """
    # Convert Date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Pretrain period: 2015-2020 (all stocks) - V7.1: TACH BIET voi test
    pretrain_mask = (df['Date'] >= pretrain_start) & (df['Date'] <= pretrain_end)
    df_pretrain = df[pretrain_mask].copy()
    
    # Fine-tune period: 2021-2024 (all stocks) - V7.1: TACH BIET voi test
    finetune_mask = (df['Date'] >= finetune_start) & (df['Date'] <= finetune_end)
    df_finetune = df[finetune_mask].copy()
    
    # Validation period: 2023-2024 (NVDA only)
    nvda_mask = df['stock_NVDA'] == 1
    val_mask = (df['Date'] >= val_start) & (df['Date'] <= val_end) & nvda_mask
    df_val = df[val_mask].copy()
    
    # Test period: 2025 (NVDA only) - OUT-OF-SAMPLE thuc su
    test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end) & nvda_mask
    df_test = df[test_mask].copy()
    
    print(f"\n{'='*60}")
    print("V7.2 Data Split by Years (No Leakage):")
    print(f"{'='*60}")
    print(f"  Pretrain (2015-2020): {len(df_pretrain)} rows (all stocks)")
    print(f"  Fine-tune (2021-2024): {len(df_finetune)} rows (all stocks)")
    print(f"  Validation (2023-2024): {len(df_val)} rows (NVDA only)")
    print(f"  Test (2025): {len(df_test)} rows (NVDA only) - OUT-OF-SAMPLE")
    
    # Prepare features and targets cho tung period
    def prepare_period(df_period):
        if len(df_period) == 0:
            return None, None, None
        X = df_period[feature_cols].values
        y_reg = df_period['future_return'].values.reshape(-1, 1)
        y_cls = df_period['signal_label'].values
        # Create sequences
        X_seq, y_reg_seq = create_sequences(X, y_reg, predictor.sequence_length)
        _, y_cls_seq = create_sequences(X, y_cls, predictor.sequence_length)
        return X_seq, y_reg_seq, y_cls_seq
    
    pretrain_data = prepare_period(df_pretrain)
    finetune_data = prepare_period(df_finetune)
    val_data = prepare_period(df_val)
    test_data = prepare_period(df_test)
    
    return {
        'pretrain': pretrain_data,
        'finetune': finetune_data,
        'validation': val_data,
        'test': test_data
    }


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    """Tao dataloaders cho training"""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    train_ds = RegressionDataset(X_train_scaled, y_train_scaled.flatten())
    test_ds = RegressionDataset(X_test_scaled, np.zeros(len(X_test)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y, X_test_scaled


def pretrain_model(input_size, train_loader, val_loader, epochs=200, lr=5e-4, device='cpu'):
    """
    Pretrain model tren 5 nam (2015-2020) - V7.1
    Train toan bo model (khong freeze)
    """
    model = LSTMRegressor(input_size=input_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_loss = float('inf')
    patience = 20
    pat = 0
    train_losses = []
    val_losses = []

    for ep in range(epochs):
        model.train()
        tr_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(Xb)
            loss = crit(out, yb.unsqueeze(1))
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)
        train_losses.append(tr_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = crit(out, yb.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)

        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}/{epochs}: Train Loss={tr_loss:.6f}, Val Loss={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            pat = 0
            torch.save(model.state_dict(), 'best_pretrain.pth')
        else:
            pat += 1
        if pat >= patience:
            print(f"  Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(torch.load('best_pretrain.pth', map_location=device))
    return model, train_losses, val_losses


def finetune_model(model, train_loader, val_loader, epochs=100, lr=1e-4, device='cpu'):
    """
    Fine-tune model tren 4 nam (2021-2024) - V7.1
    Freeze encoder (LSTM layers), chi train decoder (FC layers)
    """
    # Freeze LSTM layers (encoder)
    for name, param in model.named_parameters():
        if 'lstm' in name.lower():
            param.requires_grad = False
    
    # Chi train FC layers (decoder)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=lr)
    crit = nn.MSELoss()

    best_loss = float('inf')
    patience = 15
    pat = 0
    train_losses = []
    val_losses = []

    for ep in range(epochs):
        model.train()
        tr_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(Xb)
            loss = crit(out, yb.unsqueeze(1))
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)
        train_losses.append(tr_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = crit(out, yb.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)

        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}/{epochs}: Train Loss={tr_loss:.6f}, Val Loss={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            pat = 0
            torch.save(model.state_dict(), 'best_finetune.pth')
        else:
            pat += 1
        if pat >= patience:
            print(f"  Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(torch.load('best_finetune.pth', map_location=device))
    return model, train_losses, val_losses


def freeze_model(model):
    """
    Freeze toan bo model (khong train nua)
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # Set to eval mode
    return model


def train_model(input_size, train_loader, test_loader, epochs=200, lr=5e-4, device='cpu'):
    """Train model voi early stopping (legacy function, kept for compatibility)"""
    return pretrain_model(input_size, train_loader, test_loader, epochs, lr, device)


def detect_false_signals(signals, y_true, min_wr=MIN_WIN_RATE, min_expectancy=MIN_EXPECTANCY):
    """
    Phat hien va loai bo false signals
    False signal: win rate thap, return am, hoac expectancy < min_expectancy
    """
    buy_returns = y_true[signals == 2]
    sell_returns = y_true[signals == 0]
    
    # Buy signals
    buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
    avg_buy_return = np.mean(buy_returns) if len(buy_returns) > 0 else 0
    avg_buy_win = np.mean(buy_returns[buy_returns > 0]) if np.any(buy_returns > 0) else 0
    avg_buy_loss = np.mean(buy_returns[buy_returns <= 0]) if np.any(buy_returns <= 0) else 0
    buy_expectancy = buy_wr * avg_buy_win - (1 - buy_wr) * abs(avg_buy_loss) if len(buy_returns) > 0 else 0
    
    # Sell signals
    sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
    avg_sell_return = np.mean(sell_returns) if len(sell_returns) > 0 else 0
    avg_sell_win = np.mean(-sell_returns[sell_returns < 0]) if np.any(sell_returns < 0) else 0
    avg_sell_loss = np.mean(-sell_returns[sell_returns >= 0]) if np.any(sell_returns >= 0) else 0
    sell_expectancy = sell_wr * avg_sell_win - (1 - sell_wr) * abs(avg_sell_loss) if len(sell_returns) > 0 else 0
    
    # Check false signals (tat ca dieu kien)
    buy_is_false = (buy_wr < min_wr) or (avg_buy_return < 0) or (buy_expectancy < min_expectancy)
    sell_is_false = (sell_wr < min_wr) or (avg_sell_return > 0) or (sell_expectancy < min_expectancy)
    
    # Filter signals
    filtered_signals = signals.copy()
    if buy_is_false and len(buy_returns) > 0:
        filtered_signals[signals == 2] = 1  # Convert BUY to NO_TRADE
    if sell_is_false and len(sell_returns) > 0:
        filtered_signals[signals == 0] = 1  # Convert SELL to NO_TRADE
    
    return {
        'buy_is_false': buy_is_false,
        'sell_is_false': sell_is_false,
        'buy_wr': buy_wr,
        'sell_wr': sell_wr,
        'buy_expectancy': buy_expectancy,
        'sell_expectancy': sell_expectancy,
        'avg_buy_return': avg_buy_return,
        'avg_sell_return': avg_sell_return,
        'filtered_signals': filtered_signals
    }


def calculate_profitability_metrics(signals, y_true, stop_loss_pct=STOP_LOSS_PCT):
    """
    V7.2: Tinh cac metrics ve lai/lo thuc te voi stoploss va drawdown fix
    
    Thay doi:
    - Them stoploss gia lap: Neu daily_return < stop_loss_pct, gan bang stop_loss_pct va force exit
    - Sua tinh Drawdown: Tinh dua tren Equity Curve (Von) thay vi cumulative_returns
    """
    # Cumulative returns
    cumulative_returns = []
    position = 0  # 0: no position, 1: long, -1: short
    total_return = 0.0
    
    for i in range(len(signals)):
        if signals[i] == 2 and position != 1:  # BUY
            position = 1
        elif signals[i] == 0 and position != -1:  # SELL
            position = -1
        elif signals[i] == 1:  # NO_TRADE
            position = 0
        
        # Tinh return cho ngay nay
        if position == 1:  # Long
            daily_return = y_true[i]
        elif position == -1:  # Short
            daily_return = -y_true[i]
        else:
            daily_return = 0
        
        # V7.2: Stoploss gia lap - neu daily_return < stop_loss_pct, gan bang stop_loss_pct va force exit
        if daily_return < stop_loss_pct:
            daily_return = stop_loss_pct
            position = 0  # Force exit
        
        total_return += daily_return
        cumulative_returns.append(total_return)
    
    # Profit factor
    buy_returns = y_true[signals == 2]
    sell_returns = y_true[signals == 0]
    
    buy_profits = buy_returns[buy_returns > 0]
    buy_losses = buy_returns[buy_returns <= 0]
    sell_profits = -sell_returns[sell_returns < 0]
    sell_losses = -sell_returns[sell_returns >= 0]
    
    gross_profit = np.sum(buy_profits) + np.sum(sell_profits)
    gross_loss = abs(np.sum(buy_losses)) + abs(np.sum(sell_losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe ratio
    returns_array = np.array(cumulative_returns)
    if len(returns_array) > 1:
        returns_diff = np.diff(returns_array)
        sharpe_ratio = np.mean(returns_diff) / (np.std(returns_diff) + 1e-6) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # V7.2: Max drawdown - Tinh dua tren Equity Curve (Von) thay vi cumulative_returns
    # Von ban dau = 1.0
    # equity_curve = 1.0 + cumulative_returns
    # peak = maximum.accumulate(equity_curve)
    # drawdown = (equity_curve - peak) / peak
    if len(cumulative_returns) > 0:
        initial_capital = 1.0
        equity_curve = initial_capital + np.array(cumulative_returns)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
    else:
        max_drawdown = 0
    
    return {
        'total_return': total_return,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cumulative_returns': cumulative_returns,
        'is_profitable': total_return > 0 and profit_factor > 1.0
    }


def evaluate_thresholds_v7(y_true, y_pred, buy_percentiles=None, sell_percentiles=None,
                          buy_weight=BUY_WEIGHT, sell_weight=SELL_WEIGHT,
                          min_wr=MIN_WIN_RATE, min_expectancy=MIN_EXPECTANCY,
                          confidence_pct=CONFIDENCE_THRESHOLD_PCT):
    """
    V7: Tim threshold tot nhat voi expectancy proxy scoring
    
    Cai tien:
    - V7.1: Expectancy proxy scoring
    - V7.2: Center y_pred truoc threshold
    - V7.3: Asymmetric scoring (uu tien BUY)
    - False signal detection
    - Hybrid trade/no-trade logic
    """
    if buy_percentiles is None:
        buy_percentiles = [70, 75, 80, 85, 90]
    if sell_percentiles is None:
        sell_percentiles = [30, 25, 20, 15, 10]

    # V7.2: Center y_pred de threshold on dinh hon
    y_pred_mean = np.mean(y_pred)
    y_pred_centered = y_pred - y_pred_mean
    
    # Confidence threshold (hybrid trade/no-trade)
    confidence_threshold = np.percentile(np.abs(y_pred_centered), confidence_pct)

    best = None
    best_score = -float('inf')
    
    for bp in buy_percentiles:
        for sp in sell_percentiles:
            # Tinh threshold tu y_pred_centered
            buy_thr = np.percentile(y_pred_centered[y_pred_centered > 0], bp) if np.any(y_pred_centered > 0) else np.percentile(np.abs(y_pred_centered), bp)
            sell_thr = -np.percentile(-y_pred_centered[y_pred_centered < 0], 100 - sp) if np.any(y_pred_centered < 0) else -np.percentile(np.abs(y_pred_centered), 100 - sp)

            # Fallback neu threshold khong hop le
            if np.isnan(buy_thr) or buy_thr <= 0:
                buy_thr = np.percentile(np.abs(y_pred_centered), bp)
            if np.isnan(sell_thr) or sell_thr >= 0:
                sell_thr = -np.percentile(np.abs(y_pred_centered), 100 - sp)

            # Hybrid trade/no-trade: chi trade khi confidence cao va co expectancy
            # Tinh expectancy truoc de check
            initial_signals = np.where(y_pred_centered > buy_thr, 2, 
                                      np.where(y_pred_centered < sell_thr, 0, 1))
            
            buy_returns = y_true[initial_signals == 2]
            sell_returns = y_true[initial_signals == 0]
            
            # Tinh expectancy
            buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
            sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
            
            avg_buy_win = np.mean(buy_returns[buy_returns > 0]) if np.any(buy_returns > 0) else 0
            avg_buy_loss = np.mean(buy_returns[buy_returns <= 0]) if np.any(buy_returns <= 0) else 0
            buy_expectancy = buy_wr * avg_buy_win - (1 - buy_wr) * abs(avg_buy_loss) if len(buy_returns) > 0 else 0
            
            avg_sell_win = np.mean(-sell_returns[sell_returns < 0]) if np.any(sell_returns < 0) else 0
            avg_sell_loss = np.mean(-sell_returns[sell_returns >= 0]) if np.any(sell_returns >= 0) else 0
            sell_expectancy = sell_wr * avg_sell_win - (1 - sell_wr) * abs(avg_sell_loss) if len(sell_returns) > 0 else 0
            
            # Apply hybrid filters: confidence + expectancy
            valid_buy = (y_pred_centered > buy_thr) & (y_pred_centered > confidence_threshold) & (buy_expectancy > min_expectancy)
            valid_sell = (y_pred_centered < sell_thr) & (np.abs(y_pred_centered) > confidence_threshold) & (sell_expectancy > min_expectancy)
            
            signals = np.where(valid_buy, 2, np.where(valid_sell, 0, 1))
            
            # Recalculate sau khi filter
            buy_returns = y_true[signals == 2]
            sell_returns = y_true[signals == 0]
            
            if len(buy_returns) == 0 and len(sell_returns) == 0:
                continue
            
            buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
            sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
            
            avg_buy_win = np.mean(buy_returns[buy_returns > 0]) if np.any(buy_returns > 0) else 0
            avg_buy_loss = np.mean(buy_returns[buy_returns <= 0]) if np.any(buy_returns <= 0) else 0
            buy_expectancy = buy_wr * avg_buy_win - (1 - buy_wr) * abs(avg_buy_loss) if len(buy_returns) > 0 else 0
            
            avg_sell_win = np.mean(-sell_returns[sell_returns < 0]) if np.any(sell_returns < 0) else 0
            avg_sell_loss = np.mean(-sell_returns[sell_returns >= 0]) if np.any(sell_returns >= 0) else 0
            sell_expectancy = sell_wr * avg_sell_win - (1 - sell_wr) * abs(avg_sell_loss) if len(sell_returns) > 0 else 0
            
            # False signal detection: loai bo neu khong dat tieu chuan
            # Recalculate sau khi apply hybrid filters
            buy_returns = y_true[signals == 2]
            sell_returns = y_true[signals == 0]
            
            if len(buy_returns) == 0 and len(sell_returns) == 0:
                continue
            
            # Recalculate metrics sau khi filter
            buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
            sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
            
            avg_buy_win = np.mean(buy_returns[buy_returns > 0]) if np.any(buy_returns > 0) else 0
            avg_buy_loss = np.mean(buy_returns[buy_returns <= 0]) if np.any(buy_returns <= 0) else 0
            buy_expectancy = buy_wr * avg_buy_win - (1 - buy_wr) * abs(avg_buy_loss) if len(buy_returns) > 0 else 0
            
            avg_sell_win = np.mean(-sell_returns[sell_returns < 0]) if np.any(sell_returns < 0) else 0
            avg_sell_loss = np.mean(-sell_returns[sell_returns >= 0]) if np.any(sell_returns >= 0) else 0
            sell_expectancy = sell_wr * avg_sell_win - (1 - sell_wr) * abs(avg_sell_loss) if len(sell_returns) > 0 else 0
            
            avg_buy_return = np.mean(buy_returns) if len(buy_returns) > 0 else 0
            avg_sell_return = np.mean(sell_returns) if len(sell_returns) > 0 else 0
            
            # False signal detection: loai bo neu khong dat tieu chuan
            filtered_signals = signals.copy()
            if len(buy_returns) > 0:
                if (buy_wr < min_wr) or (avg_buy_return < 0) or (buy_expectancy < min_expectancy):
                    filtered_signals[signals == 2] = 1  # Convert to NO_TRADE
                    buy_wr = 0
                    buy_expectancy = 0
                    avg_buy_return = 0
            
            if len(sell_returns) > 0:
                if (sell_wr < min_wr) or (avg_sell_return > 0) or (sell_expectancy < min_expectancy):
                    filtered_signals[signals == 0] = 1  # Convert to NO_TRADE
                    sell_wr = 0
                    sell_expectancy = 0
                    avg_sell_return = 0
            
            # Recalculate final metrics sau khi filter false signals
            final_buy_returns = y_true[filtered_signals == 2]
            final_sell_returns = y_true[filtered_signals == 0]
            
            if len(final_buy_returns) == 0 and len(final_sell_returns) == 0:
                continue
            
            buy_wr = np.mean(final_buy_returns > 0) if len(final_buy_returns) > 0 else 0
            sell_wr = np.mean(final_sell_returns < 0) if len(final_sell_returns) > 0 else 0
            combined_wr = (buy_wr + sell_wr) / 2 if (len(final_buy_returns) > 0 or len(final_sell_returns) > 0) else 0
            
            buy_coverage = len(final_buy_returns) / len(filtered_signals)
            sell_coverage = len(final_sell_returns) / len(filtered_signals)
            
            # Recalculate expectancy sau khi filter
            # V7.2: Khoi tao buy_expectancy va sell_expectancy
            buy_expectancy = 0
            sell_expectancy = 0
            avg_buy_return = 0
            avg_sell_return = 0
            
            if len(final_buy_returns) > 0:
                avg_buy_win = np.mean(final_buy_returns[final_buy_returns > 0]) if np.any(final_buy_returns > 0) else 0
                avg_buy_loss = np.mean(final_buy_returns[final_buy_returns <= 0]) if np.any(final_buy_returns <= 0) else 0
                buy_expectancy = buy_wr * avg_buy_win - (1 - buy_wr) * abs(avg_buy_loss)
                avg_buy_return = np.mean(final_buy_returns)
            
            if len(final_sell_returns) > 0:
                avg_sell_win = np.mean(-final_sell_returns[final_sell_returns < 0]) if np.any(final_sell_returns < 0) else 0
                avg_sell_loss = np.mean(-final_sell_returns[final_sell_returns >= 0]) if np.any(final_sell_returns >= 0) else 0
                sell_expectancy = sell_wr * avg_sell_win - (1 - sell_wr) * abs(avg_sell_loss)
                avg_sell_return = np.mean(final_sell_returns)
            
            # V7.2: Siet chat thuat toan toi uu - Chi chap nhan neu ca buy_expectancy VA sell_expectancy > 0
            # Neu mot trong hai co ky vong am (<= 0), bo qua bo tham so nay
            if buy_expectancy <= 0 or sell_expectancy <= 0:
                continue  # Bo qua bo tham so nay
            
            # V7.3: Asymmetric scoring (uu tien BUY)
            score = (
                buy_weight * buy_expectancy * buy_coverage
                + sell_weight * sell_expectancy * sell_coverage
            )
            
            # Hoac don gian hon: weighted_wr * coverage
            weighted_wr = buy_wr * buy_coverage + sell_wr * sell_coverage
            simple_score = weighted_wr * (buy_coverage + sell_coverage)
            
            # Use expectancy score (tot hon)
            if score > best_score:
                best_score = score
                best = dict(
                    bp=bp, sp=sp, buy_thr=buy_thr, sell_thr=sell_thr,
                    buy_wr=buy_wr, sell_wr=sell_wr, combined_wr=combined_wr,
                    buy_expectancy=buy_expectancy, sell_expectancy=sell_expectancy,
                    buy_coverage=buy_coverage, sell_coverage=sell_coverage,
                    weighted_wr=weighted_wr,
                    coverage=(np.sum(filtered_signals != 1) / len(filtered_signals)),
                    score=score,
                    signals=filtered_signals,
                    avg_buy_return=avg_buy_return,
                    avg_sell_return=avg_sell_return
                )
    
    return best


def optimize_threshold_validation(model, X_val, y_val, scaler_X, scaler_y, device='cpu'):
    """
    Optimize threshold chi tren validation period (2023-2024)
    Custom optimization: maximize expectancy * coverage
    """
    # Predict tren validation set
    model.eval()
    with torch.no_grad():
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        preds_scaled = model(X_val_tensor).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true = y_val.flatten()
    
    # Custom optimization: maximize expectancy * coverage
    best = None
    best_score = -float('inf')
    
    buy_percentiles = [70, 75, 80, 85, 90]
    sell_percentiles = [30, 25, 20, 15, 10]
    
    # Center y_pred
    y_pred_mean = np.mean(preds)
    y_pred_centered = preds - y_pred_mean
    
    for bp in buy_percentiles:
        for sp in sell_percentiles:
            # Calculate thresholds
            buy_thr = np.percentile(y_pred_centered[y_pred_centered > 0], bp) if np.any(y_pred_centered > 0) else np.percentile(np.abs(y_pred_centered), bp)
            sell_thr = -np.percentile(-y_pred_centered[y_pred_centered < 0], 100 - sp) if np.any(y_pred_centered < 0) else -np.percentile(np.abs(y_pred_centered), 100 - sp)
            
            # Fallback
            if np.isnan(buy_thr) or buy_thr <= 0:
                buy_thr = np.percentile(np.abs(y_pred_centered), bp)
            if np.isnan(sell_thr) or sell_thr >= 0:
                sell_thr = -np.percentile(np.abs(y_pred_centered), 100 - sp)
            
            # Generate signals
            signals = np.where(y_pred_centered > buy_thr, 2,
                             np.where(y_pred_centered < sell_thr, 0, 1))
            
            buy_returns = y_true[signals == 2]
            sell_returns = y_true[signals == 0]
            
            if len(buy_returns) == 0 and len(sell_returns) == 0:
                continue
            
            # Calculate metrics
            buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
            sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
            
            avg_buy_win = np.mean(buy_returns[buy_returns > 0]) if np.any(buy_returns > 0) else 0
            avg_buy_loss = np.mean(buy_returns[buy_returns <= 0]) if np.any(buy_returns <= 0) else 0
            buy_expectancy = buy_wr * avg_buy_win - (1 - buy_wr) * abs(avg_buy_loss) if len(buy_returns) > 0 else 0
            
            avg_sell_win = np.mean(-sell_returns[sell_returns < 0]) if np.any(sell_returns < 0) else 0
            avg_sell_loss = np.mean(-sell_returns[sell_returns >= 0]) if np.any(sell_returns >= 0) else 0
            sell_expectancy = sell_wr * avg_sell_win - (1 - sell_wr) * abs(avg_sell_loss) if len(sell_returns) > 0 else 0
            
            # V7.2: Siet chat thuat toan toi uu - Chi chap nhan neu ca buy_expectancy VA sell_expectancy > 0
            # Neu mot trong hai co ky vong am (<= 0), bo qua bo tham so nay
            if buy_expectancy <= 0 or sell_expectancy <= 0:
                continue  # Bo qua bo tham so nay
            
            buy_coverage = len(buy_returns) / len(signals)
            sell_coverage = len(sell_returns) / len(signals)
            total_coverage = buy_coverage + sell_coverage
            
            # Custom score: expectancy * coverage (uu tien ca hai)
            score = (buy_expectancy * buy_coverage + sell_expectancy * sell_coverage) * total_coverage
            
            if score > best_score:
                best_score = score
                best = dict(
                    bp=bp, sp=sp, buy_thr=buy_thr, sell_thr=sell_thr,
                    buy_wr=buy_wr, sell_wr=sell_wr,
                    buy_expectancy=buy_expectancy, sell_expectancy=sell_expectancy,
                    buy_coverage=buy_coverage, sell_coverage=sell_coverage,
                    coverage=total_coverage, score=score
                )
    
    return best


def test_on_2025(model, X_test, y_test, scaler_X, scaler_y, best_thresholds, device='cpu'):
    """
    Test model tren 2025 voi thresholds da optimize
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
    
    # Apply optimized thresholds
    y_pred_mean = np.mean(preds)
    y_pred_centered = preds - y_pred_mean
    
    buy_thr = best_thresholds['buy_thr']
    sell_thr = best_thresholds['sell_thr']
    
    signals = np.where(y_pred_centered > buy_thr, 2,
                     np.where(y_pred_centered < sell_thr, 0, 1))
    
    # Calculate all metrics
    buy_returns = y_true[signals == 2]
    sell_returns = y_true[signals == 0]
    
    buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
    sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
    combined_wr = (buy_wr + sell_wr) / 2 if (len(buy_returns) > 0 or len(sell_returns) > 0) else 0
    
    avg_buy_win = np.mean(buy_returns[buy_returns > 0]) if np.any(buy_returns > 0) else 0
    avg_buy_loss = np.mean(buy_returns[buy_returns <= 0]) if np.any(buy_returns <= 0) else 0
    buy_expectancy = buy_wr * avg_buy_win - (1 - buy_wr) * abs(avg_buy_loss) if len(buy_returns) > 0 else 0
    
    avg_sell_win = np.mean(-sell_returns[sell_returns < 0]) if np.any(sell_returns < 0) else 0
    avg_sell_loss = np.mean(-sell_returns[sell_returns >= 0]) if np.any(sell_returns >= 0) else 0
    sell_expectancy = sell_wr * avg_sell_win - (1 - sell_wr) * abs(avg_sell_loss) if len(sell_returns) > 0 else 0
    
    buy_coverage = len(buy_returns) / len(signals)
    sell_coverage = len(sell_returns) / len(signals)
    
    avg_buy_return = np.mean(buy_returns) if len(buy_returns) > 0 else 0
    avg_sell_return = np.mean(sell_returns) if len(sell_returns) > 0 else 0
    
    # Profitability metrics
    profit_metrics = calculate_profitability_metrics(signals, y_true)
    
    # Model metrics
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    
    return {
        'signals': signals,
        'y_true': y_true,
        'y_pred': preds,
        'metrics': {
            'buy_wr': buy_wr,
            'sell_wr': sell_wr,
            'combined_wr': combined_wr,
            'buy_expectancy': buy_expectancy,
            'sell_expectancy': sell_expectancy,
            'buy_coverage': buy_coverage,
            'sell_coverage': sell_coverage,
            'coverage': buy_coverage + sell_coverage,
            'avg_buy_return': avg_buy_return,
            'avg_sell_return': avg_sell_return,
            'rmse': rmse,
            'mae': mae
        },
        'profitability': profit_metrics
    }


def check_overfitting_walk_forward(X_train, y_train, X_test, y_test, model, scaler_X, scaler_y, 
                                   n_windows=WALK_FORWARD_WINDOWS, device='cpu'):
    """
    Walk-forward analysis de check overfitting
    Chia test set thanh n_windows, danh gia model tren tung window
    """
    window_size = len(X_test) // n_windows
    train_losses = []
    test_losses = []
    test_metrics = []
    
    # Evaluate train loss
    model.eval()
    with torch.no_grad():
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_pred = model(X_train_tensor).cpu().numpy()
        y_train_pred = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        train_loss = np.sqrt(mean_squared_error(y_train.flatten(), y_train_pred))
    
    for i in range(n_windows):
        test_start = i * window_size
        test_end = (i + 1) * window_size if i < n_windows - 1 else len(X_test)
        
        X_test_window = X_test[test_start:test_end]
        y_test_window = y_test[test_start:test_end]
        
        # Evaluate test loss
        model.eval()
        with torch.no_grad():
            X_test_flat = X_test_window.reshape(-1, X_test_window.shape[-1])
            X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test_window.shape)
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_test_pred = model(X_test_tensor).cpu().numpy()
            y_test_pred = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            test_loss = np.sqrt(mean_squared_error(y_test_window.flatten(), y_test_pred))
        
        test_losses.append(test_loss)
        train_losses.append(train_loss)  # Same train loss for all windows
        
        # Tinh trading metrics tren window nay
        best_window = evaluate_thresholds_v7(y_test_window.flatten(), y_test_pred)
        if best_window:
            test_metrics.append({
                'combined_wr': best_window['combined_wr'],
                'buy_wr': best_window['buy_wr'],
                'sell_wr': best_window['sell_wr'],
                'coverage': best_window['coverage']
            })
        else:
            test_metrics.append({
                'combined_wr': 0,
                'buy_wr': 0,
                'sell_wr': 0,
                'coverage': 0
            })
    
    # Check overfitting: test_loss >> train_loss hoac test metrics giam dan
    avg_train_loss = np.mean(train_losses)
    avg_test_loss = np.mean(test_losses)
    overfitting_score = avg_test_loss / avg_train_loss if avg_train_loss > 0 else float('inf')
    
    # Check metrics consistency
    combined_wrs = [m['combined_wr'] for m in test_metrics]
    metrics_std = np.std(combined_wrs) if len(combined_wrs) > 0 else 0
    metrics_decreasing = all(combined_wrs[i] >= combined_wrs[i+1] for i in range(len(combined_wrs)-1)) if len(combined_wrs) > 1 else False
    
    return {
        'overfitting_score': overfitting_score,
        'is_overfitting': overfitting_score > 1.5,  # Test loss > 1.5x train loss
        'avg_train_loss': avg_train_loss,
        'avg_test_loss': avg_test_loss,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_metrics': test_metrics,
        'metrics_std': metrics_std,
        'metrics_decreasing': metrics_decreasing
    }


def main():
    parser = argparse.ArgumentParser(description='NVDA LSTM v7.2 - No-Leakage Strategy with Fixed Logic')
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(workspace_root, 'data')
    parser.add_argument('--data_dir', '-d', default=default_data_dir)
    parser.add_argument('--seq_len', '-s', type=int, default=30)
    parser.add_argument('--horizon', '-H', type=int, default=5)
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='Epochs cho pretrain (default: 200)')
    parser.add_argument('--finetune_epochs', type=int, default=100, help='Epochs cho fine-tune (default: 100)')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--pretrain_lr', type=float, default=5e-4, help='Learning rate cho pretrain (default: 5e-4)')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='Learning rate cho fine-tune (default: 1e-4)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print(f"NVDA LSTM v7.2 - No-Leakage Strategy with Fixed Logic")
    print(f"{'='*60}")
    print(f"\nV7.2 Strategy (No Data Leakage + Fixed Logic):")
    print(f"  Step 1: Pretrain trên 2015-2020 (5 năm) - TÁCH BIỆT với test")
    print(f"  Step 2: Fine-tune trên 2021-2024 (4 năm) - Freeze Encoder, TÁCH BIỆT với test")
    print(f"  Step 3: Freeze Model")
    print(f"  Step 4: Optimize Threshold trên 2023-2024 (Validation)")
    print(f"  Step 5: Test trên 2025 (Test Period) - OUT-OF-SAMPLE thực sự")
    print(f"\nV7.2 Fixes:")
    print(f"  - Loại bỏ absolute features: bb_lower, bb_middle, bb_upper, obv")
    print(f"  - Thêm Stoploss giả lập: {STOP_LOSS_PCT*100:.0f}%")
    print(f"  - Sửa Drawdown: Tính dựa trên Equity Curve (Vốn)")
    print(f"  - Siết chặt optimization: Chỉ chấp nhận nếu cả buy_expectancy VÀ sell_expectancy > 0")

    # Init predictor (chi de lay sequence_length va horizon)
    predictor = NVDA_MultiStock_Complete(sequence_length=args.seq_len, horizon=args.horizon)
    
    # ==================== Load data va them features TRUC TIEP trong v7.1 ====================
    import glob
    stocks = ['NVDA', 'AMD', 'MU', 'INTC', 'QCOM']
    all_dfs = []
    all_feature_cols = set()
    stock_quantiles = {}
    
    print(f"Loading data from {len(stocks)} stocks...")
    
    # Tim file trong data/Feature/ truoc, neu khong co thi tim trong data/
    feature_dir = os.path.join(args.data_dir, 'Feature')
    search_dirs = [feature_dir, args.data_dir] if os.path.exists(feature_dir) else [args.data_dir]
    
    for stock in stocks:
        csv_file = None
        
        # Tim file voi pattern {stock}_dss_features_*.csv
        for search_dir in search_dirs:
            pattern = os.path.join(search_dir, f"{stock}_dss_features_*.csv")
            matches = glob.glob(pattern)
            if matches:
                # Lay file moi nhat neu co nhieu file
                csv_file = max(matches, key=os.path.getmtime)
                break
        
        if csv_file is None or not os.path.exists(csv_file):
            print(f"Warning: {stock}_dss_features_*.csv not found in {search_dirs}, skipping {stock}")
            continue
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        
        # ==================== Them 5 One-Hot Encoding Features ====================
        for s in stocks:
            df[f'stock_{s}'] = 1 if s == stock else 0
        
        # Create future return target
        adj = df["Adj Close"].astype(float)
        df["future_return"] = (adj.shift(-args.horizon) / adj) - 1.0
        
        # ==================== Them 5 Sector Features ====================
        # Simulate SOX index data
        np.random.seed(42)
        df['sox_return'] = np.random.normal(0.001, 0.02, len(df))
        
        # Calculate rolling beta to SOX
        window = 20
        df['stock_vs_sox'] = df['daily_return'] - df['sox_return']
        
        # Rolling correlation
        df['rolling_corr_sox'] = df['daily_return'].rolling(window).corr(df['sox_return'])
        
        # Rolling beta
        cov_stock_sox = df['daily_return'].rolling(window).cov(df['sox_return'])
        var_sox = df['sox_return'].rolling(window).var()
        df['beta_to_sox'] = cov_stock_sox / var_sox
        
        # Sector momentum indicator
        df['sector_momentum'] = df['sox_return'].rolling(5).mean()
        
        # Calculate per-stock quantiles for labels
        r = df["future_return"].values
        r_clean = r[~np.isnan(r)]
        
        q_low = np.quantile(r_clean, 0.30)
        q_high = np.quantile(r_clean, 0.70)
        
        # Store quantiles for this stock
        stock_quantiles[stock] = {'low': q_low, 'high': q_high}
        
        print(f"{stock} quantiles - Low: {q_low:.4f}, High: {q_high:.4f}")
        
        # Create labels using per-stock quantiles
        y_cls = np.where(df["future_return"] >= q_high, 2, 
                        np.where(df["future_return"] <= q_low, 0, 1))
        df["signal_label"] = y_cls.astype(np.int64)
        
        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)
        
        # Select ONLY relative features (NO absolute prices!)
        exclude_cols = [
            "Date", "Index", "Adj Close", "Close", "Open", "High", "Low",
            "daily_return", "price_change", "future_return", "signal_label"
        ]
        
        # Check which columns exist and select features
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Ensure no absolute price features (chi loai bo cac columns chinh xac la price columns)
        absolute_price_cols = ['close', 'open', 'high', 'low']
        feature_cols = [c for c in feature_cols if c.lower() not in absolute_price_cols]
        
        print(f"  OK {stock}: {len(df)} rows, {len(feature_cols)} features (from {os.path.basename(csv_file)})")
        all_dfs.append(df)
        all_feature_cols.update(feature_cols)
    
    if not all_dfs:
        raise ValueError("No stock data files found!")
    
    # Ensure all DataFrames have the same columns
    for i, df in enumerate(all_dfs):
        for col in all_feature_cols:
            if col not in df.columns:
                df[col] = 0  # Fill missing features with 0
    
    # Combine all data
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Convert to float32 (except labels)
    label_cols = ['signal_label']
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in label_cols]
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    
    print(f"\nCombined dataset:")
    print(f"  Total rows: {len(df)}")
    print(f"  Features: {len(all_feature_cols)}")
    
    # Kiem tra date range thuc te cua dataset
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    actual_min_date = df['Date'].min()
    actual_max_date = df['Date'].max()
    
    # Normalize timezone: chuyen ve tz-naive neu la tz-aware
    if actual_min_date.tz is not None:
        actual_min_date = actual_min_date.tz_localize(None)
    if actual_max_date.tz is not None:
        actual_max_date = actual_max_date.tz_localize(None)
    
    print(f"  Date range: {actual_min_date.date()} to {actual_max_date.date()}")
    
    # Show overall label distribution
    label_counts = np.bincount(df['signal_label'].values, minlength=3)
    print(f"\nOverall Label Distribution:")
    print(f"  SELL (0): {label_counts[0]} ({label_counts[0]/len(df):.1%})")
    print(f"  NO_TRADE (1): {label_counts[1]} ({label_counts[1]/len(df):.1%})")
    print(f"  BUY (2): {label_counts[2]} ({label_counts[2]/len(df):.1%})")

    # Su dung TAT CA features co san, nhung loai bo cac features khong mong muon
    available = list(all_feature_cols)
    
    # V7.2: Loai bo cac features khong mong muon (bao gom cac features co gia tri tuyet doi)
    exclude_features = [
        "Unnamed: 0",  # Index column tu CSV
        "Volume",  # Absolute volume, da co volume_ratio va volume_sma20
        "sma50", "sma200",  # Absolute SMA values, da co price_vs_sma50 va price_vs_sma200
        "bb_lower", "bb_middle", "bb_upper",  # Absolute Bollinger Bands values
        "obv"  # Absolute On-Balance Volume
    ]
    
    # Loai bo cac features trong exclude list
    available = [f for f in available if f not in exclude_features]
    
    # Sap xep features de de doc hon
    available_sorted = sorted(available)
    
    # Phan loai features
    # One-hot: chi cac features stock_NVDA, stock_AMD, etc. (khong bao gom stock_vs_sox)
    onehot_features = [f for f in available_sorted if f.startswith('stock_') and f not in ['stock_vs_sox']]
    # Sector: cac features lien quan den sox/sector (bao gom stock_vs_sox)
    sector_features = [f for f in available_sorted if ('sox' in f.lower() or 'sector' in f.lower() or f == 'stock_vs_sox')]
    # Technical: tat ca cac features con lai
    technical_features = [f for f in available_sorted if f not in onehot_features and f not in sector_features]
    
    print(f"\n{'='*60}")
    print(f"FEATURE SUMMARY:")
    print(f"{'='*60}")
    print(f"Total features from dataset: {len(all_feature_cols)}")
    print(f"Excluded features ({len(exclude_features)}): {exclude_features}")
    print(f"Final number of features: {len(available_sorted)}")
    print(f"\nFeature breakdown:")
    print(f"  - Technical indicators: {len(technical_features)}")
    print(f"  - Sector features: {len(sector_features)}")
    print(f"  - One-hot encoding: {len(onehot_features)}")
    print(f"  - Total: {len(technical_features) + len(sector_features) + len(onehot_features)}")
    print(f"\n{'='*60}")
    print(f"DETAILED FEATURE LIST:")
    print(f"{'='*60}")
    print(f"\nTechnical Indicators ({len(technical_features)}):")
    for i, feat in enumerate(technical_features, 1):
        print(f"  {i:2d}. {feat}")
    print(f"\nSector Features ({len(sector_features)}):")
    for i, feat in enumerate(sector_features, 1):
        print(f"  {i:2d}. {feat}")
    print(f"\nOne-Hot Encoding ({len(onehot_features)}):")
    for i, feat in enumerate(onehot_features, 1):
        print(f"  {i:2d}. {feat}")
    print(f"{'='*60}")
    
    # Cap nhat available thanh sorted list
    available = available_sorted
    
    if len(available) == 0:
        raise RuntimeError("No features available in dataset. Aborting v7.2 run.")

    # Dieu chinh date ranges neu can thiet
    # Neu dataset khong co data tu 2015-2020, dieu chinh pretrain period
    pretrain_start = '2015-01-01'
    pretrain_end = '2020-12-31'
    
    # Neu dataset bat dau sau 2015, dieu chinh pretrain_start
    # Su dung tz-naive timestamp de tranh loi so sanh
    target_date_2015 = pd.to_datetime('2015-01-01')
    target_date_2020 = pd.to_datetime('2020-12-31')
    if actual_min_date > target_date_2015:
        # Su dung 2 nam dau tien cua dataset cho pretrain (neu co it nhat 2 nam)
        pretrain_start = actual_min_date.strftime('%Y-01-01')
        # Pretrain se la 2 nam dau tien, hoac den 2020 neu dataset co data den 2020
        # Normalize timezone cho pretrain_end_date
        pretrain_end_candidate = actual_min_date + pd.DateOffset(years=2)
        if pretrain_end_candidate.tz is not None:
            pretrain_end_candidate = pretrain_end_candidate.tz_localize(None)
        pretrain_end_date = min(target_date_2020, pretrain_end_candidate)
        pretrain_end = pretrain_end_date.strftime('%Y-12-31')
        print(f"\nWARNING: Dataset khong co data tu 2015. Dieu chinh pretrain period:")
        print(f"  Pretrain: {pretrain_start} to {pretrain_end}")
    
    # Split data theo V7.1 date ranges (no leakage)
    data_splits = split_data_by_years_v7_1(
        df, available, predictor,
        pretrain_start=pretrain_start,
        pretrain_end=pretrain_end,
        finetune_start='2021-01-01',
        finetune_end='2024-12-31',  # V7.1: 2024 thay vi 2025
        val_start='2023-01-01',
        val_end='2024-12-31',
        test_start='2025-01-01',
        test_end='2025-12-31'
    )

    # Check data availability va xu ly fallback neu can
    use_fallback = False
    if data_splits['pretrain'][0] is None or len(data_splits['pretrain'][0]) == 0:
        # Neu khong co pretrain data, su dung finetune data cho pretrain (fallback)
        print(f"\nWARNING: Khong co pretrain data trong khoang {pretrain_start} to {pretrain_end}")
        print(f"  Su dung finetune data (2021-2024) cho ca pretrain va finetune")
        print(f"  Strategy: Pretrain = 70% of 2021-2024, Fine-tune = 30% of 2021-2024, Test = 2025")
        
        # Su dung finetune data cho pretrain
        X_finetune, y_finetune_reg, _ = data_splits['finetune']
        if X_finetune is None or len(X_finetune) == 0:
            raise RuntimeError("No finetune data available. Cannot proceed with fallback.")
        
        # Split finetune data: 70% cho pretrain, 30% cho finetune
        finetune_split_idx = int(len(X_finetune) * 0.7)
        X_pretrain = X_finetune[:finetune_split_idx]
        y_pretrain_reg = y_finetune_reg[:finetune_split_idx]
        X_finetune_new = X_finetune[finetune_split_idx:]
        y_finetune_reg_new = y_finetune_reg[finetune_split_idx:]
        
        # Update data_splits
        data_splits['pretrain'] = (X_pretrain, y_pretrain_reg, None)
        data_splits['finetune'] = (X_finetune_new, y_finetune_reg_new, None)
        
        print(f"  Pretrain (70% of 2021-2024): {len(X_pretrain)} sequences")
        print(f"  Fine-tune (30% of 2021-2024): {len(X_finetune_new)} sequences")
        use_fallback = True
    
    if data_splits['pretrain'][0] is None or len(data_splits['pretrain'][0]) == 0:
        raise RuntimeError("No pretrain data available. Check date ranges.")
    if data_splits['finetune'][0] is None or len(data_splits['finetune'][0]) == 0:
        raise RuntimeError("No finetune data available. Check date ranges.")
    if data_splits['validation'][0] is None or len(data_splits['validation'][0]) == 0:
        raise RuntimeError("No validation data available. Check date ranges.")
    if data_splits['test'][0] is None or len(data_splits['test'][0]) == 0:
        raise RuntimeError("No test data available. Check date ranges.")

    # ==================== Step 1: Pretrain ====================
    print(f"\n{'='*60}")
    if use_fallback:
        print("Step 1: Pretrain trên 70% of 2021-2024 (Fallback - No 2015-2020 data) - V7.2")
    else:
        print("Step 1: Pretrain trên 2015-2020 (5 năm) - V7.2 No Leakage")
    print(f"{'='*60}")
    X_pretrain, y_pretrain_reg, _ = data_splits['pretrain']
    
    # Split pretrain data: 80% train, 20% val
    pretrain_split_idx = int(len(X_pretrain) * 0.8)
    X_pretrain_train = X_pretrain[:pretrain_split_idx]
    y_pretrain_train = y_pretrain_reg[:pretrain_split_idx]
    X_pretrain_val = X_pretrain[pretrain_split_idx:]
    y_pretrain_val = y_pretrain_reg[pretrain_split_idx:]
    
    print(f"  Pretrain Train: {X_pretrain_train.shape[0]} sequences")
    print(f"  Pretrain Val: {X_pretrain_val.shape[0]} sequences")
    
    # Create dataloaders cho pretrain
    pretrain_train_loader, pretrain_val_loader, scaler_X_pretrain, scaler_y_pretrain, _ = create_dataloaders(
        X_pretrain_train, y_pretrain_train, X_pretrain_val, y_pretrain_val, batch_size=args.batch_size)
    
    model_pretrained, pretrain_train_losses, pretrain_val_losses = pretrain_model(
        input_size=X_pretrain_train.shape[2],
        train_loader=pretrain_train_loader,
        val_loader=pretrain_val_loader,
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        device=device
    )
    print(f"  Pretrain completed. Best Val Loss: {min(pretrain_val_losses):.6f}")

    # ==================== Step 2: Fine-tune trên 4 năm (freeze encoder) ====================
    print(f"\n{'='*60}")
    print("Step 2: Fine-tune trên 2021-2024 (4 năm) - Freeze Encoder (V7.2 No Leakage)")
    print(f"{'='*60}")
    X_finetune, y_finetune_reg, _ = data_splits['finetune']
    
    # Split finetune data: 80% train, 20% val
    finetune_split_idx = int(len(X_finetune) * 0.8)
    X_finetune_train = X_finetune[:finetune_split_idx]
    y_finetune_train = y_finetune_reg[:finetune_split_idx]
    X_finetune_val = X_finetune[finetune_split_idx:]
    y_finetune_val = y_finetune_reg[finetune_split_idx:]
    
    print(f"  Fine-tune Train: {X_finetune_train.shape[0]} sequences")
    print(f"  Fine-tune Val: {X_finetune_val.shape[0]} sequences")
    
    # Use pretrained scalers (important for consistency)
    # Scale finetune data with pretrain scalers
    X_finetune_train_flat = X_finetune_train.reshape(-1, X_finetune_train.shape[-1])
    X_finetune_val_flat = X_finetune_val.reshape(-1, X_finetune_val.shape[-1])
    X_finetune_train_scaled = scaler_X_pretrain.transform(X_finetune_train_flat).reshape(X_finetune_train.shape)
    X_finetune_val_scaled = scaler_X_pretrain.transform(X_finetune_val_flat).reshape(X_finetune_val.shape)
    y_finetune_train_scaled = scaler_y_pretrain.transform(y_finetune_train.reshape(-1, 1))
    
    finetune_train_ds = RegressionDataset(X_finetune_train_scaled, y_finetune_train_scaled.flatten())
    finetune_val_ds = RegressionDataset(X_finetune_val_scaled, scaler_y_pretrain.transform(y_finetune_val.reshape(-1, 1)).flatten())
    finetune_train_loader = DataLoader(finetune_train_ds, batch_size=args.batch_size, shuffle=True)
    finetune_val_loader = DataLoader(finetune_val_ds, batch_size=args.batch_size, shuffle=False)
    
    model_finetuned, finetune_train_losses, finetune_val_losses = finetune_model(
        model_pretrained,
        train_loader=finetune_train_loader,
        val_loader=finetune_val_loader,
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        device=device
    )
    print(f"  Fine-tune completed. Best Val Loss: {min(finetune_val_losses):.6f}")

    # ==================== Step 3: Freeze Model ====================
    print(f"\n{'='*60}")
    print("Step 3: Freeze Model")
    print(f"{'='*60}")
    model_frozen = freeze_model(model_finetuned)
    print(f"  Model frozen (all parameters require_grad=False)")

    # ==================== Step 4: Optimize Threshold trên Validation (2023-2024) ====================
    print(f"\n{'='*60}")
    print("Step 4: Optimize Threshold trên 2023-2024 (Validation)")
    print(f"{'='*60}")
    X_val, y_val_reg, _ = data_splits['validation']
    print(f"  Validation sequences: {X_val.shape[0]}")
    
    best_thresholds = optimize_threshold_validation(
        model_frozen, X_val, y_val_reg, scaler_X_pretrain, scaler_y_pretrain, device
    )
    
    if best_thresholds is None:
        raise RuntimeError("No valid thresholds found on validation set.")
    
    print(f"  Best Thresholds:")
    print(f"    Buy percentile: {best_thresholds['bp']} -> threshold {best_thresholds['buy_thr']:.4f}")
    print(f"    Sell percentile: {best_thresholds['sp']} -> threshold {best_thresholds['sell_thr']:.4f}")
    print(f"    Buy Win Rate: {best_thresholds['buy_wr']:.1%}")
    print(f"    Sell Win Rate: {best_thresholds['sell_wr']:.1%}")
    print(f"    Buy Expectancy: {best_thresholds['buy_expectancy']:.4f}")
    print(f"    Sell Expectancy: {best_thresholds['sell_expectancy']:.4f}")
    print(f"    Coverage: {best_thresholds['coverage']:.1%}")
    print(f"    Score: {best_thresholds['score']:.4f}")

    # ==================== Step 5: Test trên 2025 ====================
    print(f"\n{'='*60}")
    print("Step 5: Test trên 2025 (Test Period)")
    print(f"{'='*60}")
    X_test, y_test_reg, _ = data_splits['test']
    print(f"  Test sequences: {X_test.shape[0]}")
    
    test_results = test_on_2025(
        model_frozen, X_test, y_test_reg, scaler_X_pretrain, scaler_y_pretrain,
        best_thresholds, device
    )
    
    print(f"\n  Test Results:")
    print(f"    Buy Win Rate: {test_results['metrics']['buy_wr']:.1%}")
    print(f"    Sell Win Rate: {test_results['metrics']['sell_wr']:.1%}")
    print(f"    Combined Win Rate: {test_results['metrics']['combined_wr']:.1%}")
    print(f"    Buy Expectancy: {test_results['metrics']['buy_expectancy']:.4f}")
    print(f"    Sell Expectancy: {test_results['metrics']['sell_expectancy']:.4f}")
    print(f"    Coverage: {test_results['metrics']['coverage']:.1%}")
    print(f"    RMSE: {test_results['metrics']['rmse']:.4f}")
    print(f"    MAE: {test_results['metrics']['mae']:.4f}")
    
    print(f"\n  Profitability Metrics:")
    print(f"    Total Return: {test_results['profitability']['total_return']:.2%}")
    print(f"    Profit Factor: {test_results['profitability']['profit_factor']:.2f}")
    print(f"    Sharpe Ratio: {test_results['profitability']['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown: {test_results['profitability']['max_drawdown']:.2%}")
    print(f"    Is Profitable: {test_results['profitability']['is_profitable']}")

    # Save artifacts
    artifact = {
        'model_state': model_frozen.state_dict(),
        'features': available,
        'num_features': len(available),
        'pretrain_metrics': {
            'best_train_loss': float(min(pretrain_train_losses)),
            'best_val_loss': float(min(pretrain_val_losses))
        },
        'finetune_metrics': {
            'best_train_loss': float(min(finetune_train_losses)),
            'best_val_loss': float(min(finetune_val_losses))
        },
        'validation_thresholds': best_thresholds,
        'test_metrics': test_results['metrics'],
        'test_profitability': test_results['profitability'],
        'config': {
            'pretrain_epochs': args.pretrain_epochs,
            'finetune_epochs': args.finetune_epochs,
            'pretrain_lr': args.pretrain_lr,
            'finetune_lr': args.finetune_lr
        }
    }
    
    # Luu artifact vao cung thu muc voi script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_path = os.path.join(script_dir, 'nvda_lstm_v7_2_artifact.pth')
    torch.save(artifact, artifact_path)
    print(f"\n{'='*60}")
    print(f"V7.2 Artifact saved: {artifact_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

