#!/usr/bin/env python3
"""
NVDA LSTM with Multi-Stock Transfer Learning
Train on NVDA + AMD + MU + INTC, test on NVDA only
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
from glob import glob

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
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
class MultiStockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- Model --------------------
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
        self.fc2 = nn.Linear(32, 1)  # Regression output
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
        self.fc2 = nn.Linear(32, 3)  # 3 classes: SELL/NO_TRADE/BUY
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


# -------------------- Multi-Stock Data Loader --------------------
class NVDA_MultiStock_LSTM:
    def __init__(self, sequence_length=60, horizon=5):
        self.sequence_length = sequence_length
        self.horizon = horizon
        
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        self.model_reg = None
        self.model_cls = None
        
        # Stock configuration
        self.stocks = ['NVDA', 'AMD', 'MU', 'INTC']
        self.stock_to_id = {stock: i for i, stock in enumerate(self.stocks)}
        
    def load_stock_data(self, csv_file):
        """Load and prepare data for a single stock"""
        stock = os.path.basename(csv_file).split('_')[0]
        df = pd.read_csv(csv_file)
        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        
        # Add stock ID as one-hot encoding
        for s in self.stocks:
            df[f'stock_{s}'] = 1 if s == stock else 0
        
        return df, stock
    
    def prepare_features(self, df):
        """Prepare features for all stocks"""
        # Create future return target
        adj = df["Adj Close"].astype(float)
        df["future_return"] = (adj.shift(-self.horizon) / adj) - 1.0
        
        # Create classification labels (using quantile for balance)
        r = df["future_return"].values
        r_clean = r[~np.isnan(r)]
        
        q_low = np.quantile(r_clean, 0.30)
        q_high = np.quantile(r_clean, 0.70)
        
        y_cls = np.where(r >= q_high, 2, np.where(r <= q_low, 0, 1))
        df["signal_label"] = y_cls.astype(int)
        
        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)
        
        # Select features (exclude target-related columns)
        exclude_cols = ["Date", "Index", "Adj Close", "Close", "daily_return", 
                       "price_change", "future_return", "signal_label"]
        
        # Check which columns exist
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        return df, feature_cols
    
    def load_multi_stock_data(self, data_dir):
        """Load and combine data from all stocks"""
        all_dfs = []
        
        print(f"📊 Loading data from {len(self.stocks)} stocks...")
        
        for stock in self.stocks:
            csv_file = os.path.join(data_dir, f"{stock}_dss_features_20251212.csv")
            
            if not os.path.exists(csv_file):
                print(f"⚠️ Warning: {csv_file} not found, skipping {stock}")
                continue
            
            df, _ = self.load_stock_data(csv_file)
            df, feature_cols = self.prepare_features(df)
            
            print(f"  ✅ {stock}: {len(df)} rows")
            all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No stock data files found!")
        
        # Combine all data
        df_combined = pd.concat(all_dfs, ignore_index=True)
        
        # Convert to float32
        numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
        df_combined[numeric_cols] = df_combined[numeric_cols].astype(np.float32)
        
        print(f"\n✅ Combined dataset:")
        print(f"  Total rows: {len(df_combined)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Date range: {df_combined['Date'].min().date()} to {df_combined['Date'].max().date()}")
        
        # Show label distribution
        label_counts = np.bincount(df_combined['signal_label'].values, minlength=3)
        print(f"\n📊 Label Distribution:")
        print(f"  SELL (0): {label_counts[0]} ({label_counts[0]/len(df_combined):.1%})")
        print(f"  NO_TRADE (1): {label_counts[1]} ({label_counts[1]/len(df_combined):.1%})")
        print(f"  BUY (2): {label_counts[2]} ({label_counts[2]/len(df_combined):.1%})")
        
        return df_combined, feature_cols
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def split_data(self, df, feature_cols, test_stock='NVDA', test_ratio=0.2):
        """
        Split data: train on all stocks except test portion of target stock
        """
        # Separate NVDA data
        nvda_mask = df['stock_NVDA'] == 1
        df_nvda = df[nvda_mask].reset_index(drop=True)
        df_others = df[~nvda_mask].reset_index(drop=True)
        
        # Split NVDA for train/test
        nvda_test_size = int(len(df_nvda) * test_ratio)
        nvda_train_idx = len(df_nvda) - nvda_test_size
        
        df_nvda_train = df_nvda.iloc[:nvda_train_idx]
        df_nvda_test = df_nvda.iloc[nvda_train_idx:]
        
        # Combine training data (NVDA train + all other stocks)
        df_train = pd.concat([df_nvda_train, df_others], ignore_index=True)
        df_test = df_nvda_test
        
        print(f"\n📊 Data Split Strategy:")
        print(f"  Train: {len(df_train)} rows (NVDA: {len(df_nvda_train)}, Others: {len(df_others)})")
        print(f"  Test: {len(df_test)} rows ({test_stock} only)")
        
        # Prepare features and targets
        X_train = df_train[feature_cols].values
        y_train_reg = df_train['future_return'].values.reshape(-1, 1)
        y_train_cls = df_train['signal_label'].values
        
        X_test = df_test[feature_cols].values
        y_test_reg = df_test['future_return'].values.reshape(-1, 1)
        y_test_cls = df_test['signal_label'].values
        
        # Create sequences
        X_train_seq, y_train_reg_seq = self.create_sequences(X_train, y_train_reg)
        X_train_seq, y_train_cls_seq = self.create_sequences(X_train, y_train_cls)
        
        X_test_seq, y_test_reg_seq = self.create_sequences(X_test, y_test_reg)
        X_test_seq, y_test_cls_seq = self.create_sequences(X_test, y_test_cls)
        
        return (X_train_seq, y_train_reg_seq, y_train_cls_seq,
                X_test_seq, y_test_reg_seq, y_test_cls_seq)
    
    def train_regression(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=1e-3):
        """Train regression model"""
        print("\n🔧 Training Regression (predict 5-day return)...")
        
        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Data loaders
        train_loader = DataLoader(MultiStockDataset(X_train_scaled, y_train_scaled.flatten()), 
                                 batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(MultiStockDataset(X_test_scaled, self.scaler_y.transform(y_test).flatten()), 
                                batch_size=batch_size, shuffle=False)
        
        # Model
        self.model_reg = LSTMRegressor(input_size=X_train.shape[2]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_reg.parameters(), lr=lr)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
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
            
            # Validate
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
                torch.save(self.model_reg.state_dict(), 'best_reg_multistock.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model_reg.load_state_dict(torch.load('best_reg_multistock.pth', map_location=device))
        print("✅ Regression training completed")
    
    def evaluate_regression(self, X_test, y_test):
        """Evaluate regression model"""
        self.model_reg.eval()
        
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        loader = DataLoader(MultiStockDataset(X_test_scaled, np.zeros(len(X_test))), 
                           batch_size=64, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(device)
                out = self.model_reg(batch_X).cpu().numpy()
                preds.append(out)
        
        y_pred_scaled = np.vstack(preds)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Metrics
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mae = np.mean(np.abs(y_pred - y_test))
        
        print(f"\n📊 Regression Evaluation (NVDA only):")
        print(f"  RMSE (return): {rmse:.4f}")
        print(f"  MAE  (return): {mae:.4f}")
        
        # Trading evaluation
        self.evaluate_trading_strategy(y_pred, y_test)
    
    def evaluate_trading_strategy(self, y_pred, y_true, threshold=0.02):
        """Evaluate trading strategy from regression predictions"""
        signals = np.where(y_pred.flatten() > threshold, 2,  # BUY
                          np.where(y_pred.flatten() < -threshold, 0, 1))  # SELL/NO_TRADE
        
        total_days = len(y_true)
        trade_days = np.sum(signals != 1)
        coverage = trade_days / total_days
        
        buy_returns = y_true.flatten()[signals == 2]
        sell_returns = y_true.flatten()[signals == 0]
        
        buy_win_rate = np.mean(buy_returns > 0) if len(buy_returns) > 0 else 0
        sell_win_rate = np.mean(sell_returns < 0) if len(sell_returns) > 0 else 0
        
        print(f"\n📊 Trading Strategy (threshold={threshold:.0%}):")
        print(f"  Coverage: {coverage:.2%} ({trade_days}/{total_days} days)")
        print(f"  Signal Distribution: BUY={np.sum(signals==2)}, SELL={np.sum(signals==0)}, NO_TRADE={np.sum(signals==1)}")
        print(f"  Win Rate - BUY: {buy_win_rate:.2%}, SELL: {sell_win_rate:.2%}")


def main():
    parser = argparse.ArgumentParser("NVDA Multi-Stock LSTM with Transfer Learning")
    parser.add_argument("--data_dir", "-d", default="data", help="Directory containing stock CSV files")
    parser.add_argument("--seq_len", "-s", type=int, default=60)
    parser.add_argument("--horizon", "-H", type=int, default=5)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--threshold", "-t", type=float, default=0.02)
    args = parser.parse_args()
    
    predictor = NVDA_MultiStock_LSTM(sequence_length=args.seq_len, horizon=args.horizon)
    
    # Load multi-stock data
    df, feature_cols = predictor.load_multi_stock_data(args.data_dir)
    
    # Split with transfer learning strategy
    (X_train, y_train_reg, y_train_cls,
     X_test, y_test_reg, y_test_cls) = predictor.split_data(df, feature_cols)
    
    print(f"\n📊 Final Data Shapes:")
    print(f"  Train: {X_train.shape} (sequences × timesteps × features)")
    print(f"  Test: {X_test.shape}")
    
    # Train and evaluate regression
    predictor.train_regression(X_train, y_train_reg, X_test, y_test_reg, 
                              epochs=args.epochs, batch_size=args.batch_size)
    predictor.evaluate_regression(X_test, y_test_reg)
    
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
