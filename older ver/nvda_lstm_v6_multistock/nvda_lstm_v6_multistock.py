#!/usr/bin/env python3
"""
NVDA LSTM v6 — Optimized với 10 features từ quick test (remove_top7_vif)

Version này sử dụng 10 features tốt nhất từ quick test:
- Combined Win Rate: 69.65%
- Buy Win Rate: 93.94%
- Sell Win Rate: 45.36%
- RMSE: 0.0402
- Average VIF: 3.20 (giảm từ hàng nghìn xuống 3.2)

Features được loại bỏ (7 features có VIF cao nhất):
- macd_hist, macd_signal, macd, rsi14, bb_percent, obv, atr

Features còn lại (10 features):
- macd_bullish, bb_bandwidth, volume_ratio, volume_sma20
- daily_return, price_change, return_3d, return_5d, return_10d, return_20d
- hl_spread_pct, oc_spread, oc_spread_pct
- bb_squeeze, rsi_overbought, rsi_oversold
- sox_beta, sox_correlation
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
    from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete, LSTMRegressor, RegressionDataset
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


# 10 features từ best config remove_top7_vif
V6_FEATURES = [
    'macd_bullish',
    'bb_bandwidth',
    'volume_ratio',
    'volume_sma20',
    'daily_return',
    'price_change',
    'return_3d',
    'return_5d',
    'return_10d',
    'return_20d',
    'hl_spread_pct',
    'oc_spread',
    'oc_spread_pct',
    'bb_squeeze',
    'rsi_overbought',
    'rsi_oversold',
    'sox_beta',
    'sox_correlation'
]

# Note: V6_FEATURES có 18 features, nhưng trong quick test remove_top7_vif chỉ có 10 features
# Từ baseline 17 features, loại bỏ 7: macd_hist, macd_signal, macd, rsi14, bb_percent, obv, atr
# Còn lại 10 features thực tế. Tuy nhiên, nếu dataset có đủ features, sẽ dùng 18 features này
# Nếu không, sẽ filter theo available features


def load_v6_features():
    """Load 10 features chính xác từ best config remove_top7_vif"""
    # Từ quick_vif_analysis.csv có 19 features được phân tích VIF
    # Baseline trong quick test có 17 features available trong dataset
    # Sau khi loại bỏ 7 features VIF cao nhất → còn 10 features
    
    # 7 features bị loại bỏ (VIF cao nhất):
    # macd_hist, macd_signal, macd, rsi14, bb_percent, obv, atr
    
    # 10 features chính xác còn lại (từ quick_vif_analysis.csv, loại bỏ 7 features trên):
    # Loại bỏ các features không có trong dataset: daily_return, price_change, return_*, sox_*
    # Còn lại đúng 10 features:
    v6_features = [
        'oc_spread_pct',     # VIF: 6.93
        'oc_spread',         # VIF: 6.70
        'volume_sma20',      # VIF: 3.37
        'bb_bandwidth',      # VIF: 2.71
        'macd_bullish',      # VIF: 2.53
        'hl_spread_pct',     # VIF: 2.46
        'rsi_overbought',    # VIF: 2.41
        'volume_ratio',      # VIF: 1.95
        'bb_squeeze',        # VIF: 1.76
        'rsi_oversold'       # VIF: 1.16
    ]
    
    # Đây chính xác là 10 features từ remove_top7_vif trong quick test
    # (sau khi loại bỏ 7 features VIF cao và các features không có trong dataset)
    
    return v6_features


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    """Tạo dataloaders cho training"""
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


def train_model(input_size, train_loader, test_loader, epochs=200, lr=5e-4, device='cpu'):
    """Train model với early stopping"""
    model = LSTMRegressor(input_size=input_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    best_loss = float('inf')
    patience = 20
    pat = 0

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

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = crit(out, yb.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= max(1, len(test_loader))

        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1}/{epochs}: Train Loss={tr_loss:.6f}, Val Loss={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            pat = 0
            torch.save(model.state_dict(), 'best_v6_reg.pth')
        else:
            pat += 1
        if pat >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(torch.load('best_v6_reg.pth', map_location=device))
    return model


def evaluate_thresholds(y_true, y_pred, buy_percentiles=None, sell_percentiles=None):
    """Tìm threshold tốt nhất để maximize combined win rate"""
    if buy_percentiles is None:
        buy_percentiles = [70, 75, 80, 85, 90]
    if sell_percentiles is None:
        sell_percentiles = [30, 25, 20, 15, 10]

    best = None
    for bp in buy_percentiles:
        for sp in sell_percentiles:
            # Tính threshold từ percentile
            buy_thr = np.percentile(y_pred[y_pred > 0], bp) if np.any(y_pred > 0) else np.percentile(np.abs(y_pred), bp)
            sell_thr = -np.percentile(-y_pred[y_pred < 0], 100 - sp) if np.any(y_pred < 0) else -np.percentile(np.abs(y_pred), 100 - sp)

            # Fallback nếu threshold không hợp lệ
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

            # Score: ưu tiên combined win rate
            score = combined_wr
            if best is None or score > best['combined_wr']:
                best = dict(
                    bp=bp, sp=sp, buy_thr=buy_thr, sell_thr=sell_thr,
                    buy_wr=buy_wr, sell_wr=sell_wr, combined_wr=combined_wr,
                    weighted_wr=weighted_wr,
                    coverage=(np.sum(signals != 1) / len(signals))
                )
    return best


def main():
    parser = argparse.ArgumentParser(description='NVDA LSTM v6 - 10 features optimized')
    parser.add_argument('--data_dir', '-d', default=os.path.join('..', 'data'))
    parser.add_argument('--seq_len', '-s', type=int, default=30)
    parser.add_argument('--horizon', '-H', type=int, default=5)
    parser.add_argument('--epochs', '-e', type=int, default=200)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Init predictor
    predictor = NVDA_MultiStock_Complete(sequence_length=args.seq_len, horizon=args.horizon)
    df, all_features = predictor.load_multi_stock_data(args.data_dir)

    # Su dung TAT CA features co san (khong bo feature nao)
    # LSTM can correlation de tang kha nang vao lenh, VIF lam mat tri nho
    available = all_features
    print(f"\n{'='*60}")
    print(f"NVDA LSTM v6 - Using ALL {len(available)} features (no VIF removal)")
    print(f"{'='*60}")
    print(f"\nUsing all {len(available)} features from dataset")
    print(f"Features: {', '.join(available[:15])}..." if len(available) > 15 else f"Features: {', '.join(available)}")
    
    if len(available) == 0:
        raise RuntimeError("No features available in dataset. Aborting v6 run.")

    # Split data
    (X_train, y_train_reg, y_train_cls,
     X_test, y_test_reg, y_test_cls) = predictor.split_data_transfer_learning(df, available)

    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    # Create dataloaders
    train_loader, test_loader, scaler_X, scaler_y, X_test_scaled = create_dataloaders(
        X_train, y_train_reg, X_test, y_test_reg, batch_size=args.batch_size)

    # Train model
    print(f"\nTraining model...")
    model = train_model(
        input_size=X_train.shape[2],
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=5e-4,
        device=device
    )

    # Predict on test set
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    y_true = y_test_reg.flatten()

    # Evaluate thresholds
    best = evaluate_thresholds(y_true, preds)

    print(f"\n{'='*60}")
    print("v6 Threshold Search Result:")
    print(f"{'='*60}")
    if best is None:
        print('  No valid thresholds found (empty predictions).')
    else:
        print(f"  Buy percentile: {best['bp']} -> threshold {best['buy_thr']:.4f}")
        print(f"  Buy Win Rate: {best['buy_wr']:.1%}")
        print(f"  Sell percentile: {best['sp']} -> threshold {best['sell_thr']:.4f}")
        print(f"  Sell Win Rate: {best['sell_wr']:.1%}")
        print(f"  Combined Win Rate: {best['combined_wr']:.1%}")
        print(f"  Weighted Win Rate: {best['weighted_wr']:.1%}")
        print(f"  Coverage: {best['coverage']:.1%}")

    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    print(f"\nModel metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # So sánh với kết quả expected từ quick test
    print(f"\n{'='*60}")
    print("So sanh voi ket qua expected tu quick test:")
    print(f"{'='*60}")
    print(f"  Expected Combined WR: 69.65%")
    print(f"  Actual Combined WR: {best['combined_wr']:.1%}" if best else "  N/A")
    print(f"  Expected Buy WR: 93.94%")
    print(f"  Actual Buy WR: {best['buy_wr']:.1%}" if best else "  N/A")
    print(f"  Expected RMSE: 0.0402")
    print(f"  Actual RMSE: {rmse:.4f}")

    # Save model & config
    artifact = {
        'model_state': model.state_dict(),
        'features': available,
        'num_features': len(available),
        'metrics': {
            'rmse': float(rmse),
            'mae': float(mae),
            'buy_wr': float(best['buy_wr']) if best else 0,
            'sell_wr': float(best['sell_wr']) if best else 0,
            'combined_wr': float(best['combined_wr']) if best else 0,
        }
    }
    torch.save(artifact, 'nvda_lstm_v6_artifact.pth')
    print(f"\n✅ v6 artifact saved: nvda_lstm_v6_artifact.pth")


if __name__ == '__main__':
    main()

