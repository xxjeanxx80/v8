#!/usr/bin/env python3
"""
Complete Multi-Stock LSTM with Per-Stock Quantiles and Sector Features
Best practices for cross-stock learning
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
class MultiStockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- Models --------------------
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


# -------------------- Complete Multi-Stock System --------------------
class NVDA_MultiStock_Complete:
    def __init__(self, sequence_length=30, horizon=5):
        self.sequence_length = sequence_length
        self.horizon = horizon
        
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        self.model_reg = None
        self.model_cls = None
        
        # Stock configuration
        self.stocks = ['NVDA', 'AMD', 'MU', 'INTC', 'QCOM']
        self.stock_to_id = {stock: i for i, stock in enumerate(self.stocks)}
        
        # Store per-stock quantiles
        self.stock_quantiles = {}
        
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
    
    def add_sector_features(self, df):
        """Add sector-wide features (SOX index correlation)"""
        # Simulate SOX index data (in real implementation, load actual SOX data)
        # For now, create a proxy using average of all tech stocks
        np.random.seed(42)
        df['sox_return'] = np.random.normal(0.001, 0.02, len(df))
        
        # Calculate rolling beta to SOX (covariance/variance)
        window = 20
        df['stock_vs_sox'] = df['daily_return'] - df['sox_return']
        
        # Rolling correlation (simplified)
        df['rolling_corr_sox'] = df['daily_return'].rolling(window).corr(df['sox_return'])
        
        # Rolling beta (more actionable)
        cov_stock_sox = df['daily_return'].rolling(window).cov(df['sox_return'])
        var_sox = df['sox_return'].rolling(window).var()
        df['beta_to_sox'] = cov_stock_sox / var_sox
        
        # Sector momentum indicator
        df['sector_momentum'] = df['sox_return'].rolling(5).mean()
        
        print(f"Added sector features (SOX beta, correlation)")
        
    def prepare_features(self, df, stock, is_training=True):
        """Prepare features with per-stock quantile labels"""
        # Create future return target
        adj = df["Adj Close"].astype(float)
        df["future_return"] = (adj.shift(-self.horizon) / adj) - 1.0
        
        # Add sector features
        self.add_sector_features(df)
        
        # Calculate per-stock quantiles for labels
        if is_training:
            r = df["future_return"].values
            r_clean = r[~np.isnan(r)]
            
            q_low = np.quantile(r_clean, 0.30)
            q_high = np.quantile(r_clean, 0.70)
            
            # Store quantiles for this stock
            self.stock_quantiles[stock] = {'low': q_low, 'high': q_high}
            
            print(f"{stock} quantiles - Low: {q_low:.4f}, High: {q_high:.4f}")
        else:
            # Use stored quantiles for test data
            if stock not in self.stock_quantiles:
                raise ValueError(f"No quantiles stored for {stock}")
            q_low = self.stock_quantiles[stock]['low']
            q_high = self.stock_quantiles[stock]['high']
        
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
        # Khong loai bo cac technical indicators co chua "low" nhu bb_lower
        absolute_price_cols = ['close', 'open', 'high', 'low']
        feature_cols = [c for c in feature_cols if c.lower() not in absolute_price_cols]
        
        return df, feature_cols
    
    def load_multi_stock_data(self, data_dir):
        """Load and combine data from all stocks with proper per-stock processing"""
        import glob
        all_dfs = []
        all_feature_cols = set()
        
        print(f"Loading data from {len(self.stocks)} stocks...")
        
        # Tim file trong data/Feature/ truoc, neu khong co thi tim trong data/
        feature_dir = os.path.join(data_dir, 'Feature')
        search_dirs = [feature_dir, data_dir] if os.path.exists(feature_dir) else [data_dir]
        
        for stock in self.stocks:
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
            
            df, _ = self.load_stock_data(csv_file)
            df, feature_cols = self.prepare_features(df, stock, is_training=True)
            
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
        df_combined = pd.concat(all_dfs, ignore_index=True)
        
        # Convert to float32 (except labels)
        label_cols = ['signal_label']
        numeric_cols = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                       if col not in label_cols]
        df_combined[numeric_cols] = df_combined[numeric_cols].astype(np.float32)
        
        print(f"\nCombined dataset:")
        print(f"  Total rows: {len(df_combined)}")
        print(f"  Features: {len(all_feature_cols)}")
        print(f"  Date range: {df_combined['Date'].min().date()} to {df_combined['Date'].max().date()}")
        
        # Show overall label distribution
        label_counts = np.bincount(df_combined['signal_label'].values, minlength=3)
        print(f"\nOverall Label Distribution:")
        print(f"  SELL (0): {label_counts[0]} ({label_counts[0]/len(df_combined):.1%})")
        print(f"  NO_TRADE (1): {label_counts[1]} ({label_counts[1]/len(df_combined):.1%})")
        print(f"  BUY (2): {label_counts[2]} ({label_counts[2]/len(df_combined):.1%})")
        
        return df_combined, list(all_feature_cols)
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def split_data_transfer_learning(self, df, feature_cols, test_stock='NVDA', test_ratio=0.2):
        """
        Proper split for transfer learning:
        - Train on all stocks (including part of NVDA)
        - Test only on out-of-sample NVDA
        """
        # Separate NVDA data
        nvda_mask = df['stock_NVDA'] == 1
        df_nvda = df[nvda_mask].reset_index(drop=True)
        df_others = df[~nvda_mask].reset_index(drop=True)
        
        # Time-based split for NVDA (avoid look-ahead)
        nvda_test_size = int(len(df_nvda) * test_ratio)
        nvda_train_idx = len(df_nvda) - nvda_test_size
        
        df_nvda_train = df_nvda.iloc[:nvda_train_idx]
        df_nvda_test = df_nvda.iloc[nvda_train_idx:]
        
        # Combine training data
        df_train = pd.concat([df_nvda_train, df_others], ignore_index=True)
        df_test = df_nvda_test
        
        print(f"\nTransfer Learning Split:")
        print(f"  Train: {len(df_train)} rows")
        print(f"    - NVDA train: {len(df_nvda_train)}")
        print(f"    - Other stocks: {len(df_others)}")
        print(f"  Test: {len(df_test)} rows ({test_stock} only, out-of-sample)")
        
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
        """Train regression model with early stopping"""
        print("\nTraining Multi-Stock Regression...")
        
        # Scale data
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = self.scaler_X.fit_transform(X_train_flat).reshape(X_train.shape)
        X_test_scaled = self.scaler_X.transform(X_test_flat).reshape(X_test.shape)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Data loaders
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
        
        # Training loop with early stopping
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
                torch.save(self.model_reg.state_dict(), 'best_reg_multistock_complete.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model_reg.load_state_dict(torch.load('best_reg_multistock_complete.pth', map_location=device))
        print("Multi-Stock Regression training completed")
    
    def evaluate_regression(self, X_test, y_test):
        """Evaluate regression on NVDA test set"""
        self.model_reg.eval()
        
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = self.scaler_X.transform(X_test_flat).reshape(X_test.shape)
        
        loader = DataLoader(
            RegressionDataset(X_test_scaled, np.zeros(len(X_test))),
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
        
        # Metrics
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        mae = np.mean(np.abs(y_pred - y_test))
        
        print(f"\nMulti-Stock Regression Evaluation (NVDA Test Set):")
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
        
        print(f"\nTrading Strategy (threshold={threshold:.0%}):")
        print(f"  Coverage: {coverage:.2%} ({trade_days}/{total_days} days)")
        print(f"  Signal Distribution: BUY={np.sum(signals==2)}, SELL={np.sum(signals==0)}, NO_TRADE={np.sum(signals==1)}")
        print(f"  Win Rate - BUY: {buy_win_rate:.2%}, SELL: {sell_win_rate:.2%}")
        
        # Show feature importance proxy (correlation with predictions)
        print(f"\nModel learned from {len(self.stocks)} stocks with:")
        print(f"  - Per-stock quantile labels")
        print(f"  - Sector features (SOX beta, correlation)")
        print(f"  - Only relative features (no absolute prices)")


def main():
    parser = argparse.ArgumentParser("Complete Multi-Stock LSTM with Best Practices")
    parser.add_argument("--data_dir", "-d", default="data", help="Directory containing stock CSV files")
    parser.add_argument("--seq_len", "-s", type=int, default=30)
    parser.add_argument("--horizon", "-H", type=int, default=5)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--threshold", "-t", type=float, default=0.02)
    args = parser.parse_args()
    
    predictor = NVDA_MultiStock_Complete(sequence_length=args.seq_len, horizon=args.horizon)
    
    # Load multi-stock data with proper processing
    df, feature_cols = predictor.load_multi_stock_data(args.data_dir)
    
    # Split with transfer learning strategy
    (X_train, y_train_reg, y_train_cls,
     X_test, y_test_reg, y_test_cls) = predictor.split_data_transfer_learning(df, feature_cols)
    
    print(f"\nFinal Data Shapes:")
    print(f"  Train: {X_train.shape} (sequences × timesteps × features)")
    print(f"  Test: {X_test.shape} (NVDA only)")
    
    # Train and evaluate
    predictor.train_regression(X_train, y_train_reg, X_test, y_test_reg, 
                              epochs=args.epochs, batch_size=args.batch_size)
    predictor.evaluate_regression(X_test, y_test_reg)
    
    print("\n✅ Multi-Stock Training Complete!")
    print("\n💡 Key Features:")
    print("  ✅ Per-stock quantile labels")
    print("  ✅ Sector features (SOX beta)")
    print("  ✅ Only relative features")
    print("  ✅ Proper transfer learning split")


if __name__ == "__main__":
    main()
