#!/usr/bin/env python3
"""
Enhanced NVDA LSTM with more data and better features
Works with current data, ready for multi-stock extension
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
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
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- Enhanced Models --------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
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
    def __init__(self, input_size, hidden_size=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 3)
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


# -------------------- Enhanced Predictor --------------------
class NVDA_LSTM_Enhanced:
    def __init__(self, sequence_length=30, horizon=5):  # Reduced seq length for more samples
        self.sequence_length = sequence_length
        self.horizon = horizon
        
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        self.model_reg = None
        self.model_cls = None
        
    def load_data(self, csv_file):
        """Load and enhance data"""
        df = pd.read_csv(csv_file)
        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        
        # Create future return target
        adj = df["Adj Close"].astype(float)
        df["future_return"] = (adj.shift(-self.horizon) / adj) - 1.0
        
        # Enhanced features
        self.add_enhanced_features(df)
        
        # Create balanced classification labels
        r = df["future_return"].values
        r_clean = r[~np.isnan(r)]
        
        q_low = np.quantile(r_clean, 0.30)
        q_high = np.quantile(r_clean, 0.70)
        
        y_cls = np.where(r >= q_high, 2, np.where(r <= q_low, 0, 1))
        df["signal_label"] = y_cls.astype(int)
        
        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} rows")
        print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Label distribution
        label_counts = np.bincount(df['signal_label'].values, minlength=3)
        print(f"\n📊 Label Distribution:")
        print(f"  SELL (0): {label_counts[0]} ({label_counts[0]/len(df):.1%})")
        print(f"  NO_TRADE (1): {label_counts[1]} ({label_counts[1]/len(df):.1%})")
        print(f"  BUY (2): {label_counts[2]} ({label_counts[2]/len(df):.1%})")
        
        return df
    
    def add_enhanced_features(self, df):
        """Add additional features for better prediction"""
        # Multi-timeframe momentum
        for period in [3, 7, 14, 30]:
            df[f'momentum_{period}'] = df['Adj Close'].pct_change(period)
        
        # Volatility regime
        df['volatility_5d'] = df['daily_return'].rolling(5).std()
        df['volatility_20d'] = df['daily_return'].rolling(20).std()
        df['vol_regime'] = df['volatility_5d'] / df['volatility_20d']
        
        # Trend strength
        df['trend_5d'] = (df['Adj Close'] > df['Adj Close'].rolling(5).mean()).astype(int)
        df['trend_20d'] = (df['Adj Close'] > df['Adj Close'].rolling(20).mean()).astype(int)
        
        # Price position
        df['price_position_20'] = (df['Adj Close'] - df['Adj Close'].rolling(20).min()) / \
                                  (df['Adj Close'].rolling(20).max() - df['Adj Close'].rolling(20).min())
        
        # Volume-price interaction
        df['volume_price_trend'] = df['volume_ratio'] * np.sign(df['daily_return'])
        
        # RSI zones
        df['rsi_zone'] = np.where(df['rsi14'] > 70, 2, 
                                 np.where(df['rsi14'] < 30, 0, 1))
        
        # MACD divergence indicator
        df['macd_divergence'] = np.where(
            (df['macd'] > df['macd_signal']) & (df['macd'] > df['macd'].shift(5)), 1, 0
        )
        
        print(f"🔧 Added enhanced features")
        
    def prepare_data(self, df):
        """Prepare features and targets"""
        exclude_cols = ["Date", "Index", "Adj Close", "Close", "daily_return", 
                       "price_change", "future_return", "signal_label"]
        
        self.feature_columns = [c for c in df.columns if c not in exclude_cols]
        
        X = df[self.feature_columns].values.astype(np.float32)
        y_reg = df['future_return'].values.astype(np.float32).reshape(-1, 1)
        y_cls = df['signal_label'].values.astype(np.int64)
        
        return X, y_reg, y_cls
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def split_data(self, X_seq, y_seq, split=0.8):
        """Time-based split"""
        idx = int(len(X_seq) * split)
        return X_seq[:idx], X_seq[idx:], y_seq[:idx], y_seq[idx:]
    
    def train_regression(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=1e-3):
        """Train regression model with early stopping"""
        print("\n🔧 Training Regression (5-day return prediction)...")
        
        # Scale data
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = self.scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = self.scaler_X.transform(X_test_flat).reshape(X_test.shape)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Data loaders - use FloatTensor for regression targets
        class RegressionDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X)
                self.y = torch.FloatTensor(y)
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        train_loader = DataLoader(
            RegressionDataset(X_train_scaled, y_train_scaled.flatten()),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            RegressionDataset(X_test_scaled, self.scaler_y.transform(y_test).flatten()),
            batch_size=batch_size, shuffle=False
        )
        
        # Model
        self.model_reg = LSTMRegressor(input_size=X_train.shape[2]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_reg.parameters(), lr=lr)
        
        # Training with early stopping
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model_reg.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model_reg(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model_reg.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.model_reg(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model_reg.state_dict(), 'best_reg_enhanced.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model_reg.load_state_dict(torch.load('best_reg_enhanced.pth', map_location=device))
        print("✅ Regression training completed")
    
    def evaluate_regression(self, X_test, y_test):
        """Evaluate regression with trading metrics"""
        self.model_reg.eval()
        
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = self.scaler_X.transform(X_test_flat).reshape(X_test.shape)
        
        loader = DataLoader(
            StockDataset(X_test_scaled, np.zeros(len(X_test))),
            batch_size=64, shuffle=False
        )
        
        preds = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(device)
                out = self.model_reg(batch_X).cpu().numpy()
                preds.append(out)
        
        y_pred_scaled = np.vstack(preds)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Basic metrics
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mae = np.mean(np.abs(y_pred - y_test))
        
        print(f"\n📊 Regression Evaluation:")
        print(f"  RMSE (return): {rmse:.4f}")
        print(f"  MAE  (return): {mae:.4f}")
        
        # Trading evaluation
        self.evaluate_trading_performance(y_pred, y_test)
    
    def evaluate_trading_performance(self, y_pred, y_true):
        """Comprehensive trading evaluation"""
        print(f"\n📈 Trading Performance Analysis:")
        
        # Prediction distribution
        print(f"\n  Prediction Distribution:")
        print(f"    Min: {np.min(y_pred):.4f}")
        print(f"    Max: {np.max(y_pred):.4f}")
        print(f"    Mean: {np.mean(y_pred):.4f}")
        print(f"    Std: {np.std(y_pred):.4f}")
        
        # True return distribution
        print(f"\n  True Return Distribution:")
        print(f"    Min: {np.min(y_true):.4f}")
        print(f"    Max: {np.max(y_true):.4f}")
        print(f"    Mean: {np.mean(y_true):.4f}")
        print(f"    Std: {np.std(y_true):.4f}")
        
        # Test multiple thresholds
        thresholds = [0.005, 0.01, 0.015, 0.02]
        
        for thr in thresholds:
            signals = np.where(y_pred.flatten() > thr, 2,  # BUY
                             np.where(y_pred.flatten() < -thr, 0, 1))  # SELL/NO_TRADE
            
            total_days = len(y_true)
            trade_days = np.sum(signals != 1)
            coverage = trade_days / total_days
            
            if trade_days > 0:
                buy_returns = y_true.flatten()[signals == 2]
                sell_returns = y_true.flatten()[signals == 0]
                
                buy_win_rate = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
                sell_win_rate = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
                
                avg_buy_return = np.mean(buy_returns) if len(buy_returns) > 0 else 0
                avg_sell_return = np.mean(sell_returns) if len(sell_returns) > 0 else 0
                
                print(f"\n  Threshold {thr:.1%}:")
                print(f"    Coverage: {coverage:.1%} ({trade_days} trades)")
                print(f"    BUY: Win Rate {buy_win_rate:.1%}, Avg Return {avg_buy_return:.2%}")
                print(f"    SELL: Win Rate {sell_win_rate:.1%}, Avg Return {avg_sell_return:.2%}")


def main():
    parser = argparse.ArgumentParser("Enhanced NVDA LSTM Trading System")
    parser.add_argument("--data", "-d", default="NVDA_dss_features_20251212.csv")
    parser.add_argument("--seq_len", "-s", type=int, default=30)
    parser.add_argument("--horizon", "-H", type=int, default=5)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    args = parser.parse_args()
    
    predictor = NVDA_LSTM_Enhanced(sequence_length=args.seq_len, horizon=args.horizon)
    
    print("📊 Loading and preparing data...")
    df = predictor.load_data(args.data)
    X, y_reg, y_cls = predictor.prepare_data(df)
    
    # Create sequences
    X_seq, y_seq_reg = predictor.create_sequences(X, y_reg)
    
    print(f"\n📊 Data Shapes:")
    print(f"  Sequences: {X_seq.shape}")
    print(f"  Features per timestep: {X_seq.shape[2]}")
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X_seq, y_seq_reg)
    print(f"\n📊 Train/Test Split: {len(X_train)}/{len(X_test)}")
    
    # Train and evaluate
    predictor.train_regression(X_train, y_train, X_test, y_test, 
                              epochs=args.epochs, batch_size=args.batch_size)
    predictor.evaluate_regression(X_test, y_test)
    
    print("\n✅ Done.")
    print("\n💡 To extend with multi-stock data:")
    print("1. Download data for AMD, MU, INTC using download_stocks.py")
    print("2. Generate features for each stock")
    print("3. Run nvda_lstm_multistock.py for transfer learning")


if __name__ == "__main__":
    main()
