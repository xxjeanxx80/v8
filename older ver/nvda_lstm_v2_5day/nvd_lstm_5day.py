#!/usr/bin/env python3
"""
NVDA 5-day ahead prediction with PyTorch LSTM
- Regression: predict 5-day ahead return
- Classification: SELL / NO_TRADE / BUY based on 5-day ahead return threshold
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------- Reproducibility --------------------
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- Dataset --------------------
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        # y có thể là float (regression) hoặc int (classification)
        if y.dtype in (np.int32, np.int64):
            self.y = torch.LongTensor(y)
        else:
            self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- Models --------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMClassifier3(nn.Module):
    """3-class classifier: 0=SELL, 1=NO_TRADE, 2=BUY (logits output)"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 3)  # 3 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # logits
        return out


# -------------------- Predictor --------------------
class NVDA_LSTM_5Day:
    def __init__(self, sequence_length=60, horizon=5, threshold=0.02):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.threshold = threshold

        self.scaler_X = MinMaxScaler()
        self.scaler_y_reg = MinMaxScaler()  # regression target scaler

        self.feature_columns = None
        self.model_reg = None
        self.model_cls = None

    def load_data(self, csv_file):
        df = pd.read_csv(csv_file)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

        # --- Feature columns: tránh leak giá trực tiếp ---
        exclude_cols = ["Date", "Index", "Adj Close", "Close", "daily_return", "price_change"]
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]

        # --- tạo target 5-day ahead return từ Adj Close ---
        adj = df["Adj Close"].astype(float)

        # future_return_5d[t] = AdjClose[t+H]/AdjClose[t] - 1
        df["future_return"] = (adj.shift(-self.horizon) / adj) - 1.0

        # --- tạo label 3 lớp từ future_return ---
        # Use quantile-based labels for balanced classes
        r = df["future_return"].values
        r_clean = r[~np.isnan(r)]
        
        # Calculate quantiles for balanced 3-class distribution
        q_low = np.quantile(r_clean, 0.30)
        q_high = np.quantile(r_clean, 0.70)
        
        # 0 = SELL (bottom 30%), 1 = NO_TRADE (middle 40%), 2 = BUY (top 30%)
        y_cls = np.where(r >= q_high, 2, np.where(r <= q_low, 0, 1))
        df["signal_label"] = y_cls.astype(int)
        
        print(f"📊 Label Distribution (quantile-based):")
        print(f"  SELL (0): {np.sum(y_cls == 0)} ({np.mean(y_cls == 0):.1%})")
        print(f"  NO_TRADE (1): {np.sum(y_cls == 1)} ({np.mean(y_cls == 1):.1%})")
        print(f"  BUY (2): {np.sum(y_cls == 2)} ({np.mean(y_cls == 2):.1%})")
        print(f"  Thresholds: SELL ≤ {q_low:.4f}, BUY ≥ {q_high:.4f}")

        # drop last H rows (vì future_return NaN)
        df = df.iloc[:-self.horizon].copy().reset_index(drop=True)

        print(f"✅ Loaded {len(df)} rows")
        if "Date" in df.columns:
            print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"🧠 Features used: {len(self.feature_columns)}")
        print(f"🎯 Horizon: {self.horizon} days | Threshold: {self.threshold*100:.2f}%")

        return df

    def prepare_data(self, df):
        X = df[self.feature_columns].values.astype(np.float32)

        # Regression target: future_return (shape Nx1)
        y_reg = df["future_return"].values.reshape(-1, 1).astype(np.float32)

        # Classification target: signal_label (shape N,)
        y_cls = df["signal_label"].values.astype(np.int64)

        X_scaled = self.scaler_X.fit_transform(X)
        y_reg_scaled = self.scaler_y_reg.fit_transform(y_reg)

        return X_scaled, y_reg_scaled, y_cls

    def create_sequences_reg(self, X, y_reg_scaled):
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y_reg_scaled[i])  # target ở thời điểm i (đã là 5-day ahead return của i)
        return np.array(X_seq), np.array(y_seq)

    def create_sequences_cls(self, X, y_cls):
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y_cls[i])  # label ở i
        return np.array(X_seq), np.array(y_seq)

    def split_data(self, X_seq, y_seq, split=0.8):
        idx = int(len(X_seq) * split)
        X_train, X_test = X_seq[:idx], X_seq[idx:]
        y_train, y_test = y_seq[:idx], y_seq[idx:]
        print(f"📊 Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    # -------------------- Train Regression --------------------
    def train_regression(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=1e-3):
        print("\n🔧 Training Regression (predict 5-day return)...")

        train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        self.model_reg = LSTMRegressor(input_size=X_train.shape[2]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_reg.parameters(), lr=lr)

        best_val = float("inf")
        patience, pc = 15, 0

        for epoch in range(epochs):
            self.model_reg.train()
            tr_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = self.model_reg(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(train_loader)

            self.model_reg.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = self.model_reg(bx)
                    val_loss += criterion(out, by).item()
            val_loss /= len(test_loader)

            if val_loss < best_val:
                best_val = val_loss
                pc = 0
                torch.save(self.model_reg.state_dict(), "best_reg_5d.pth")
            else:
                pc += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {tr_loss:.6f} Val Loss: {val_loss:.6f}")

            if pc >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.model_reg.load_state_dict(torch.load("best_reg_5d.pth", map_location=device))
        print("✅ Regression training done.")

    def evaluate_regression(self, X_test, y_test):
        self.model_reg.eval()
        loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

        preds = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(device)
                out = self.model_reg(bx).cpu().numpy()
                preds.append(out)

        y_pred_scaled = np.vstack(preds)
        y_pred = self.scaler_y_reg.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y_reg.inverse_transform(y_test)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        print("\n📊 Regression (5-day return) Evaluation:")
        print(f"  RMSE (return): {rmse:.4f}")
        print(f"  MAE  (return): {mae:.4f}")
        return y_true.flatten(), y_pred.flatten()

    # -------------------- Train Classification --------------------
    def train_classification(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=1e-3):
        print("\n🔧 Training Classification (SELL/NO_TRADE/BUY)...")

        train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        self.model_cls = LSTMClassifier3(input_size=X_train.shape[2]).to(device)

        # Remove class weights to let model learn naturally
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_cls.parameters(), lr=lr)

        best_val = float("inf")
        patience, pc = 15, 0

        for epoch in range(epochs):
            self.model_cls.train()
            tr_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = self.model_cls(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(train_loader)

            self.model_cls.eval()
            val_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = self.model_cls(bx)
                    loss = criterion(logits, by)
                    val_loss += loss.item()

                    pred = torch.argmax(logits, dim=1)
                    correct += (pred == by).sum().item()
                    total += by.size(0)

            val_loss /= len(test_loader)
            val_acc = correct / max(total, 1)

            if val_loss < best_val:
                best_val = val_loss
                pc = 0
                torch.save(self.model_cls.state_dict(), "best_cls_5d.pth")
            else:
                pc += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {tr_loss:.6f} Val Loss: {val_loss:.6f} Val Acc: {val_acc:.4f}")

            if pc >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.model_cls.load_state_dict(torch.load("best_cls_5d.pth", map_location=device))
        print("✅ Classification training done.")

    def evaluate_classification(self, X_test, y_test):
        self.model_cls.eval()
        loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

        preds = []
        trues = []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                logits = self.model_cls(bx).cpu().numpy()
                preds.append(logits)
                trues.append(by.numpy())

        logits = np.vstack(preds)
        y_pred = np.argmax(logits, axis=1)
        y_true = np.concatenate(trues)

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")

        print("\n📊 Classification Evaluation (0=SELL,1=NO_TRADE,2=BUY):")
        print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Macro F1 : {f1m:.4f}")
        print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print("\nReport:\n", classification_report(y_true, y_pred, target_names=["SELL", "NO_TRADE", "BUY"]))
        return y_true, y_pred

    # -------------------- Trading signal from regression --------------------
    def regression_to_signal(self, pred_return, threshold=None):
        """Map predicted 5-day return -> SELL/NO_TRADE/BUY"""
        thr = self.threshold if threshold is None else threshold
        if pred_return > thr:
            return 2
        if pred_return < -thr:
            return 0
        return 1
    
    def evaluate_regression_trading(self, X_test, y_test, threshold=None):
        """
        Evaluate regression model as trading strategy.
        
        Args:
            X_test: Test sequences
            y_test: True 5-day returns
            threshold: Signal threshold (default uses self.threshold)
            
        Returns:
            Trading metrics dictionary
        """
        print(f"\n📊 Trading Strategy Evaluation (threshold={threshold or self.threshold:.2%}):")
        
        # Get predictions
        self.model_reg.eval()
        loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(device)
                out = self.model_reg(bx).cpu().numpy()
                preds.append(out)
        
        y_pred_scaled = np.vstack(preds)
        y_pred = self.scaler_y_reg.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y_reg.inverse_transform(y_test)
        
        # Print prediction distribution
        print(f"\n📈 Prediction Distribution:")
        print(f"  Min: {np.min(y_pred):.4f}")
        print(f"  Max: {np.max(y_pred):.4f}")
        print(f"  Mean: {np.mean(y_pred):.4f}")
        print(f"  Std: {np.std(y_pred):.4f}")
        print(f"  25th percentile: {np.percentile(y_pred, 25):.4f}")
        print(f"  75th percentile: {np.percentile(y_pred, 75):.4f}")
        
        # Print true return distribution for comparison
        print(f"\n📈 True Return Distribution:")
        print(f"  Min: {np.min(y_true):.4f}")
        print(f"  Max: {np.max(y_true):.4f}")
        print(f"  Mean: {np.mean(y_true):.4f}")
        print(f"  Std: {np.std(y_true):.4f}")
        print(f"  25th percentile: {np.percentile(y_true, 25):.4f}")
        print(f"  75th percentile: {np.percentile(y_true, 75):.4f}")
        
        # Convert to signals
        thr = self.threshold if threshold is None else threshold
        pred_signals = np.array([self.regression_to_signal(p, thr) for p in y_pred.flatten()])
        
        # Also test percentile-based threshold for desired coverage
        if threshold is None:  # Only show this for default threshold
            for coverage_target in [0.2, 0.3, 0.4]:
                abs_pred = np.abs(y_pred.flatten())
                perc_thr = np.quantile(abs_pred, 1 - coverage_target)
                perc_signals = np.where(abs_pred >= perc_thr, 
                                        np.where(y_pred.flatten() > 0, 2, 0), 1)
                perc_coverage = np.sum(perc_signals != 1) / len(y_true)
                print(f"\n  Percentile threshold for {coverage_target:.0%} coverage: {perc_thr:.4f}")
                print(f"    Actual coverage: {perc_coverage:.2%}")
        
        # Calculate trading metrics
        total_days = len(y_true)
        trade_days = np.sum(pred_signals != 1)  # Days with BUY or SELL
        coverage = trade_days / total_days
        
        # Calculate returns for each signal
        buy_returns = y_true.flatten()[pred_signals == 2]
        sell_returns = y_true.flatten()[pred_signals == 0]
        
        # Win rates
        buy_win_rate = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
        sell_win_rate = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0  # Win if negative return for SELL
        overall_win_rate = np.mean((pred_signals == 2) & (y_true.flatten() > 0)) + \
                          np.mean((pred_signals == 0) & (y_true.flatten() < 0))
        
        # Average returns
        avg_buy_return = np.mean(buy_returns) if len(buy_returns) > 0 else 0
        avg_sell_return = np.mean(sell_returns) if len(sell_returns) > 0 else 0
        
        # Profit factor (gross profit / gross loss)
        buy_profits = buy_returns[buy_returns > 0]
        buy_losses = buy_returns[buy_returns <= 0]
        sell_profits = -sell_returns[sell_returns < 0]  # Profit from short position
        sell_losses = -sell_returns[sell_returns >= 0]
        
        gross_profit = np.sum(buy_profits) + np.sum(sell_profits)
        gross_loss = -np.sum(buy_losses) - np.sum(sell_losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Signal distribution
        signal_counts = np.bincount(pred_signals, minlength=3)
        signal_names = ['SELL', 'NO_TRADE', 'BUY']
        
        # Print metrics
        print(f"  Coverage: {coverage:.2%} ({trade_days}/{total_days} trading days)")
        print(f"  Signal Distribution:")
        for i, name in enumerate(signal_names):
            print(f"    {name}: {signal_counts[i]} ({signal_counts[i]/total_days:.2%})")
        print(f"  Win Rates:")
        print(f"    BUY: {buy_win_rate:.2%} ({len(buy_returns)} trades)")
        print(f"    SELL: {sell_win_rate:.2%} ({len(sell_returns)} trades)")
        print(f"  Average Returns:")
        print(f"    BUY: {avg_buy_return:.2%}")
        print(f"    SELL: {avg_sell_return:.2%}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        
        # Calculate overall strategy return
        strategy_returns = np.where(pred_signals == 2, y_true.flatten(),
                                   np.where(pred_signals == 0, -y_true.flatten(), 0))
        total_return = np.sum(strategy_returns)
        print(f"  Total Strategy Return: {total_return:.2%}")
        
        metrics = {
            'coverage': coverage,
            'buy_win_rate': buy_win_rate,
            'sell_win_rate': sell_win_rate,
            'avg_buy_return': avg_buy_return,
            'avg_sell_return': avg_sell_return,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'signal_counts': signal_counts
        }
        
        return metrics


def main():
    parser = argparse.ArgumentParser("NVDA 5-day LSTM (Regression + 3-class Trading)")
    parser.add_argument("--data", "-d", default="NVDA_dss_features_20251212.csv")
    parser.add_argument("--seq_len", "-s", type=int, default=60)
    parser.add_argument("--horizon", "-H", type=int, default=5)
    parser.add_argument("--threshold", "-T", type=float, default=0.02, help="Threshold for signal e.g. 0.02 for 2 percent")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--model", "-m", choices=["regression", "classification", "both"], default="both")
    args = parser.parse_args()

    predictor = NVDA_LSTM_5Day(sequence_length=args.seq_len, horizon=args.horizon, threshold=args.threshold)

    print("📊 Loading and preparing data...")
    df = predictor.load_data(args.data)
    X_scaled, y_reg_scaled, y_cls = predictor.prepare_data(df)

    # --- Regression sequences ---
    X_seq_reg, y_seq_reg = predictor.create_sequences_reg(X_scaled, y_reg_scaled)
    X_train_r, X_test_r, y_train_r, y_test_r = predictor.split_data(X_seq_reg, y_seq_reg)

    # --- Classification sequences ---
    X_seq_cls, y_seq_cls = predictor.create_sequences_cls(X_scaled, y_cls)
    X_train_c, X_test_c, y_train_c, y_test_c = predictor.split_data(X_seq_cls, y_seq_cls)

    if args.model in ["regression", "both"]:
        predictor.train_regression(X_train_r, y_train_r, X_test_r, y_test_r, epochs=args.epochs, batch_size=args.batch_size)
        y_true_r, y_pred_r = predictor.evaluate_regression(X_test_r, y_test_r)
        
        # Trading strategy evaluation
        trading_metrics = predictor.evaluate_regression_trading(X_test_r, y_test_r, threshold=args.threshold)
        
        # demo mapping regression prediction -> signal
        last_pred = y_pred_r[-1]
        sig = predictor.regression_to_signal(last_pred, args.threshold)
        sig_name = ["SELL", "NO_TRADE", "BUY"][sig]
        print(f"\n🧾 Example signal from regression (pred 5d return={last_pred:.4f}): {sig_name}")

    if args.model in ["classification", "both"]:
        predictor.train_classification(X_train_c, y_train_c, X_test_c, y_test_c, epochs=args.epochs, batch_size=args.batch_size)
        predictor.evaluate_classification(X_test_c, y_test_c)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
