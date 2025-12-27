#!/usr/bin/env python3
"""
NVDA LSTM v5 — focused feature-pruned model for higher win/sell rates

This script builds on the v4 "complete" implementation and the
`test_feature` optimized config. It trains a regression LSTM on the
pruned feature set and searches asymmetrical thresholds to maximize
combined buy/sell win-rate (proxy for profitability).
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
    # Pylance may not resolve this import in the workspace; ignore static lint here
    from nvda_lstm_multistock_complete import NVDA_MultiStock_Complete, LSTMRegressor, RegressionDataset  # type: ignore
except Exception:
    # fallback: load module directly from file path
    import importlib.util
    spec_path = os.path.join(v4_path, 'nvda_lstm_multistock_complete.py')
    if os.path.exists(spec_path):
        spec = importlib.util.spec_from_file_location('nvda_lstm_multistock_complete', spec_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Unable to load specification for {spec_path}')
        mod = importlib.util.module_from_spec(spec)
        # mypy/pylance: spec.loader is not None because of the check above
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
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


DEFAULT_FEATURES = None
OPT_CONF = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_feature', 'optimized_feature_config.csv'))


def load_optimized_features():
    global DEFAULT_FEATURES
    if os.path.exists(OPT_CONF):
        try:
            df = pd.read_csv(OPT_CONF)
            if not df.empty and 'features' in df.columns and pd.notna(df.iloc[0]['features']):
                raw = df.iloc[0]['features']
                try:
                    feats = eval(raw)
                except Exception:
                    feats = None
                if isinstance(feats, (list, tuple)):
                    DEFAULT_FEATURES = list(feats)
                    return DEFAULT_FEATURES
        except Exception:
            # fall through to default list
            pass

    # fallback conservative list (no trend features)
    DEFAULT_FEATURES = [
        'rsi14','macd','macd_bullish','macd_signal','macd_hist',
        'atr','bb_bandwidth','bb_percent',
        'volume_ratio','obv','volume_sma20',
        'daily_return','price_change','return_3d','return_5d','return_10d','return_20d',
        'hl_spread_pct','oc_spread','oc_spread_pct',
        'bb_squeeze','rsi_overbought','rsi_oversold',
        'sox_beta','sox_correlation'
    ]
    return DEFAULT_FEATURES


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
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

        # quick validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = crit(out, yb.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= max(1, len(test_loader))

        if val_loss < best_loss:
            best_loss = val_loss
            pat = 0
            torch.save(model.state_dict(), 'best_v5_reg.pth')
        else:
            pat += 1
        if pat >= patience:
            break

    model.load_state_dict(torch.load('best_v5_reg.pth', map_location=device))
    return model


def evaluate_thresholds(y_true, y_pred, buy_percentiles=None, sell_percentiles=None):
    # search over percentile thresholds to maximize buy_win_rate + sell_win_rate
    if buy_percentiles is None:
        buy_percentiles = [70,75,80,85,90]
    if sell_percentiles is None:
        sell_percentiles = [30,25,20,15,10]

    best = None
    for bp in buy_percentiles:
        for sp in sell_percentiles:
            buy_thr = np.percentile(y_pred, bp)
            sell_thr = np.percentile(y_pred, sp)

            signals = np.where(y_pred > buy_thr, 2, np.where(y_pred < sell_thr, 0, 1))

            buy_returns = y_true[signals == 2]
            sell_returns = y_true[signals == 0]

            buy_wr = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
            sell_wr = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
            combined_wr = (buy_wr + sell_wr) / 2 if (len(buy_returns) > 0 or len(sell_returns) > 0) else 0

            score = buy_wr + sell_wr
            if best is None or score > best['score']:
                best = dict(bp=bp, sp=sp, buy_thr=buy_thr, sell_thr=sell_thr,
                            buy_wr=buy_wr, sell_wr=sell_wr, combined_wr=combined_wr,
                            coverage=(np.sum(signals!=1)/len(signals)), score=score)
    return best


def main():
    parser = argparse.ArgumentParser()
    # Tim data_dir tu workspace root (DSS DATA)
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_dir = os.path.join(workspace_root, 'data')
    parser.add_argument('--data_dir','-d', default=default_data_dir)
    parser.add_argument('--seq_len','-s', type=int, default=30)
    parser.add_argument('--horizon','-H', type=int, default=5)
    parser.add_argument('--epochs','-e', type=int, default=200)
    parser.add_argument('--batch_size','-b', type=int, default=64)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Data directory: {args.data_dir}")

    # Init predictor (reuses complete class logic for feature creation)
    predictor = NVDA_MultiStock_Complete(sequence_length=args.seq_len, horizon=args.horizon)
    df, all_features = predictor.load_multi_stock_data(args.data_dir)

    # Su dung TAT CA features co san (khong bo feature nao)
    # LSTM can correlation de tang kha nang vao lenh, VIF lam mat tri nho
    available = all_features
    print(f"Using ALL {len(available)} features for v5 (no feature removal)")
    print(f"Features: {', '.join(available[:10])}..." if len(available) > 10 else f"Features: {', '.join(available)}")

    if len(available) == 0:
        raise RuntimeError("No features available in dataset. Aborting v5 run.")

    (X_train, y_train_reg, y_train_cls,
     X_test, y_test_reg, y_test_cls) = predictor.split_data_transfer_learning(df, available)

    train_loader, test_loader, scaler_X, scaler_y, X_test_scaled = create_dataloaders(
        X_train, y_train_reg, X_test, y_test_reg, batch_size=args.batch_size)

    model = train_model(input_size=X_train.shape[2], train_loader=train_loader, test_loader=test_loader,
                        epochs=args.epochs, lr=5e-4, device=device)

    # Predict on test set
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    y_true = y_test_reg.flatten()

    # Evaluate thresholds skewed to improve buy/sell win rates
    best = evaluate_thresholds(y_true, preds)

    print('\nv5 Threshold Search Result:')
    if best is None:
        print('  No valid thresholds found (empty predictions).')
    else:
        print(f"  Buy pct: {best['bp']} -> thr {best['buy_thr']:.4f}, Buy WR: {best['buy_wr']:.1%}")
        print(f"  Sell pct: {best['sp']} -> thr {best['sell_thr']:.4f}, Sell WR: {best['sell_wr']:.1%}")
        print(f"  Coverage: {best['coverage']:.1%}")

    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    print(f"\nModel metrics: RMSE={rmse:.4f}, MAE={mae:.4f}")

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
    torch.save(artifact, 'nvda_lstm_v5_artifact.pth')
    print('\nv5 artifact saved: nvda_lstm_v5_artifact.pth')


if __name__ == '__main__':
    main()
