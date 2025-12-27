#!/usr/bin/env python3
"""
NVDA 2-Stage LSTM Trading Model
Stage 1: Volume + Volatility → Trade / NoTrade
Stage 2: Price + Momentum → Buy / Sell (only on Trade days)
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)  # For classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------- Models --------------------
class LSTMStage1(nn.Module):
    """Stage 1: Predict Trade/NoTrade based on volume & volatility"""
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 2)  # 2 classes: NoTrade/Trade
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


class LSTMStage2(nn.Module):
    """Stage 2: Predict Buy/Sell based on price & momentum"""
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 2)  # 2 classes: Sell/Buy
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


# -------------------- Feature Engineering --------------------
def add_volume_features(df):
    """Add advanced volume features"""
    # Volume SMA ratios
    df['volume_sma_10'] = df['Volume'].rolling(10).mean()
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio_10'] = df['Volume'] / df['volume_sma_10']
    df['volume_ratio_20'] = df['Volume'] / df['volume_sma_20']
    
    # Relative volume (percentile)
    df['relative_volume'] = df['Volume'].rolling(20).rank(pct=True)
    
    # OBV momentum
    df['obv'] = (np.where(df['Adj Close'] > df['Adj Close'].shift(), df['Volume'], 
                         np.where(df['Adj Close'] < df['Adj Close'].shift(), -df['Volume'], 0))).cumsum()
    df['obv_sma'] = df['obv'].rolling(10).mean()
    df['obv_roc'] = df['obv'] / df['obv'].shift(5) - 1  # 5-day rate of change
    
    # Volume Price Trend (VPT)
    df['vpt'] = (df['Volume'] * df['daily_return']).cumsum()
    df['vpt_sma'] = df['vpt'].rolling(10).mean()
    
    # Volume acceleration
    df['volume_accel'] = df['Volume'].rolling(3).mean().pct_change(2)
    
    return df


def add_volatility_features(df):
    """Add advanced volatility features"""
    # ATR and its ratios
    df['atr_ratio'] = df['atr'] / df['Adj Close']
    df['atr_sma'] = df['atr'].rolling(10).mean()
    df['atr_ratio_sma'] = df['atr_ratio'].rolling(10).mean()
    
    # Bollinger Band width percentile
    df['bb_width_pct'] = df['bb_bandwidth'].rolling(20).rank(pct=True)
    
    # Price range volatility
    df['price_range'] = (df['High'] - df['Low']) / df['Open']
    df['range_sma'] = df['price_range'].rolling(10).mean()
    df['range_ratio'] = df['price_range'] / df['range_sma']
    
    # GARCH-like volatility estimate
    df['volatility'] = df['daily_return'].rolling(20).std()
    df['volatility_rank'] = df['volatility'].rolling(60).rank(pct=True)
    
    # Volatility regime
    df['vol_regime'] = np.where(df['volatility_rank'] > 0.7, 2,  # High vol
                               np.where(df['volatility_rank'] < 0.3, 0, 1))  # Low/Medium vol
    
    return df


def add_price_momentum_features(df):
    """Add price and momentum features"""
    # Multiple timeframe returns
    for period in [3, 5, 10, 20]:
        df[f'return_{period}d'] = df['Adj Close'].pct_change(period)
        df[f'return_{period}d_rank'] = df[f'return_{period}d'].rolling(60).rank(pct=True)
    
    # RSI divergence
    df['rsi_divergence'] = np.where(
        (df['rsi14'] > 70) & (df['Adj Close'] > df['sma50']), 1,  # Bullish divergence
        np.where((df['rsi14'] < 30) & (df['Adj Close'] < df['sma50']), -1, 0)  # Bearish divergence
    )
    
    # MACD momentum
    df['macd_accel'] = df['macd'].diff()
    df['macd_hist_accel'] = df['macd_hist'].diff()
    
    # Price vs moving averages
    df['price_vs_sma20'] = (df['Adj Close'] / df['sma50'] - 1) * 100  # Use sma50 as medium-term
    df['price_vs_sma50'] = (df['Adj Close'] / df['sma50'] - 1) * 100
    df['price_vs_sma200'] = (df['Adj Close'] / df['sma200'] - 1) * 100
    
    # Price position in Bollinger Bands
    df['bb_position'] = (df['Adj Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Momentum strength
    df['momentum_strength'] = np.sqrt(df['return_5d']**2 + df['return_10d']**2 + df['return_20d']**2)
    
    # Breakout confirmation features
    # Price breakout: current price > 20-day high from yesterday
    df['price_breakout'] = (df['Adj Close'] > df['Adj Close'].rolling(20).max().shift(1)).astype(int)
    
    # Volume breakout: volume > 1.5x average volume
    df['vol_breakout'] = (df['volume_ratio_20'] > 1.5).astype(int)
    
    # Breakout confirmed: both price and volume breakouts
    df['breakout_confirmed'] = (df['price_breakout'] & df['vol_breakout']).astype(int)
    
    # Breakout strength: combine price move and volume
    df['breakout_strength'] = df['price_breakout'] * df['volume_ratio_20']
    
    return df


# -------------------- 2-Stage Predictor --------------------
class NVDA_LSTM_2Stage:
    def __init__(self, sequence_length=60, horizon=5):
        self.sequence_length = sequence_length
        self.horizon = horizon
        
        self.scaler_X1 = MinMaxScaler()  # Stage 1 features
        self.scaler_X2 = MinMaxScaler()  # Stage 2 features
        
        self.model_stage1 = None
        self.model_stage2 = None
        
        # Feature groups
        self.volume_features = [
            'Volume', 'volume_ratio_10', 'volume_ratio_20', 'relative_volume',
            'obv_roc', 'vpt_sma', 'volume_accel'
        ]
        
        self.volatility_features = [
            'atr_ratio', 'atr_ratio_sma', 'bb_width_pct', 'range_ratio',
            'volatility', 'volatility_rank'
        ]
        
        self.price_features = [
            'Adj Close', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
            'rsi14', 'macd', 'macd_hist', 'price_vs_sma20', 'price_vs_sma50',
            'price_vs_sma200', 'bb_position', 'momentum_strength',
            'price_breakout', 'vol_breakout', 'breakout_confirmed', 'breakout_strength'
        ]
    
    def load_data(self, csv_file):
        """Load and enhance data with trading features"""
        df = pd.read_csv(csv_file)
        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        
        # Add advanced features
        print("🔧 Adding volume features...")
        df = add_volume_features(df)
        print("🔧 Adding volatility features...")
        df = add_volatility_features(df)
        print("🔧 Adding price & momentum features...")
        df = add_price_momentum_features(df)
        
        # Create target labels
        adj = df["Adj Close"].astype(float)
        df["future_return"] = (adj.shift(-self.horizon) / adj) - 1.0
        
        # Stage 1 target: Trade/NoTrade based on volatility threshold
        vol_threshold = df['volatility'].quantile(0.6)  # Top 40% volatile days
        df['stage1_label'] = np.where(df['volatility'] > vol_threshold, 1, 0).astype(np.int64)  # 1=Trade, 0=NoTrade
        
        # Stage 2 target: Buy/Sell (balanced classes)
        # Use median split for balanced classes (50/50)
        return_threshold = df['future_return'].median()
        df['stage2_label'] = np.where(df['future_return'] > return_threshold, 1, 0).astype(np.int64)  # 1=Buy, 0=Sell
        
        # Drop NaN rows and ensure correct dtypes
        df = df.dropna().reset_index(drop=True)
        
        # Convert all numeric columns except labels to float32 to avoid type mismatch
        label_cols = ['stage1_label', 'stage2_label']
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in label_cols]
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        
        print(f"✅ Loaded {len(df)} rows")
        print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"📊 Stage 1 - Trade days: {np.sum(df['stage1_label'])} ({np.mean(df['stage1_label']):.1%})")
        print(f"📊 Stage 2 - Buy signals: {np.sum(df['stage2_label'])} ({np.mean(df['stage2_label']):.1%})")
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for both stages"""
        # Stage 1 features: Volume + Volatility
        stage1_features = self.volume_features + self.volatility_features
        X1 = df[stage1_features].values.astype(np.float32)
        y1 = df['stage1_label'].values.astype(np.int64)
        
        # Stage 2 features: Price + Momentum
        stage2_features = self.price_features
        X2 = df[stage2_features].values.astype(np.float32)
        y2 = df['stage2_label'].values.astype(np.int64)
        
        # Scale features
        X1_scaled = self.scaler_X1.fit_transform(X1)
        X2_scaled = self.scaler_X2.fit_transform(X2)
        
        return X1_scaled, y1, X2_scaled, y2
    
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
    
    def train_stage1(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=1e-3):
        """Train Stage 1 model"""
        print("\n🔧 Training Stage 1: Trade/NoTrade...")
        
        train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        
        self.model_stage1 = LSTMStage1(input_size=X_train.shape[2]).to(device)
        
        # Class weights for imbalance
        class_counts = np.bincount(y_train)
        pos_weight = class_counts[0] / class_counts[1]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device))
        optimizer = optim.Adam(self.model_stage1.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model_stage1.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model_stage1(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model_stage1.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.model_stage1(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predicted = torch.argmax(outputs, dim=1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
            
            val_loss /= len(test_loader)
            val_acc = correct / total
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model_stage1.state_dict(), 'best_stage1.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model_stage1.load_state_dict(torch.load('best_stage1.pth', map_location=device))
        print("✅ Stage 1 training completed")
    
    def train_stage2(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=1e-3):
        """Train Stage 2 model"""
        print("\n🔧 Training Stage 2: Buy/Sell...")
        
        train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
        
        self.model_stage2 = LSTMStage2(input_size=X_train.shape[2]).to(device)
        
        # Class weights
        class_counts = np.bincount(y_train)
        pos_weight = class_counts[0] / class_counts[1]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device))
        optimizer = optim.Adam(self.model_stage2.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model_stage2.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model_stage2(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model_stage2.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.model_stage2(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predicted = torch.argmax(outputs, dim=1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
            
            val_loss /= len(test_loader)
            val_acc = correct / total
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model_stage2.state_dict(), 'best_stage2.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model_stage2.load_state_dict(torch.load('best_stage2.pth', map_location=device))
        print("✅ Stage 2 training completed")
    
    def evaluate_3stage(self, X1_test, y1_test, X2_test, y2_test, df_test):
        """Evaluate the complete 3-stage system with rule-based entry gate"""
        print("\n📊 3-Stage System Evaluation (with Entry Gate):")
        
        # Stage 1 predictions
        self.model_stage1.eval()
        loader1 = DataLoader(StockDataset(X1_test, y1_test), batch_size=64, shuffle=False)
        
        stage1_preds = []
        with torch.no_grad():
            for batch_X, _ in loader1:
                batch_X = batch_X.to(device)
                outputs = self.model_stage1(batch_X)
                predicted = torch.argmax(outputs, dim=1)
                stage1_preds.extend(predicted.cpu().numpy())
        
        stage1_preds = np.array(stage1_preds)
        
        # Stage 2 predictions (only on Trade days)
        trade_indices = np.where(stage1_preds == 1)[0]
        
        if len(trade_indices) > 0:
            X2_trade = X2_test[trade_indices]
            y2_trade = y2_test[trade_indices]
            
            # Check label distribution in Stage 2 test set
            trade_label_counts = np.bincount(y2_trade, minlength=2)
            print(f"\n  Stage 2 Test Label Distribution:")
            print(f"    Sell (0): {trade_label_counts[0]} ({trade_label_counts[0]/len(y2_trade):.1%})")
            print(f"    Buy (1): {trade_label_counts[1]} ({trade_label_counts[1]/len(y2_trade):.1%})")
            
            if trade_label_counts[0] == len(y2_trade) or trade_label_counts[1] == len(y2_trade):
                print(f"  ⚠️ Warning: Stage 2 test set has only one class! Accuracy metrics are misleading.")
            
            self.model_stage2.eval()
            loader2 = DataLoader(StockDataset(X2_trade, y2_trade), batch_size=64, shuffle=False)
            
            stage2_preds = []
            with torch.no_grad():
                for batch_X, _ in loader2:
                    batch_X = batch_X.to(device)
                    outputs = self.model_stage2(batch_X)
                    predicted = torch.argmax(outputs, dim=1)
                    stage2_preds.extend(predicted.cpu().numpy())
            
            stage2_preds = np.array(stage2_preds)
            
            # Stage 3: Apply entry gate (rule-based)
            final_preds = np.zeros(len(y1_test))
            
            # Get breakout confirmation for test period
            # Note: df_test should have the same index as the test sequences
            test_start_idx = len(df_test) - len(y1_test)
            breakout_confirmed_test = df_test['breakout_confirmed'].iloc[test_start_idx:].values
            
            # Debug: Check breakout confirmation distribution
            breakout_counts = np.bincount(breakout_confirmed_test.astype(int), minlength=2)
            print(f"\n  Breakout Confirmation Distribution:")
            print(f"    Not Confirmed (0): {breakout_counts[0]} ({breakout_counts[0]/len(breakout_confirmed_test):.1%})")
            print(f"    Confirmed (1): {breakout_counts[1]} ({breakout_counts[1]/len(breakout_confirmed_test):.1%})")
            
            # Apply 3-stage logic
            for i, trade_idx in enumerate(trade_indices):
                # Stage 1: Trade day?
                if stage1_preds[trade_idx] == 0:
                    final_preds[trade_idx] = 0  # NoTrade
                else:
                    # Stage 2: Direction prediction
                    direction = stage2_preds[i]  # 0=Sell, 1=Buy
                    
                    # Stage 3: Entry gate - breakout confirmation?
                    if breakout_confirmed_test[trade_idx] == 1:
                        # Entry confirmed - keep the direction
                        final_preds[trade_idx] = 2 if direction == 1 else 1  # 2=Buy, 1=Sell
                    else:
                        # Entry not confirmed - force NoTrade
                        final_preds[trade_idx] = 0
            
            # Calculate metrics
            print(f"  Trade Coverage: {len(trade_indices)}/{len(y1_test)} ({len(trade_indices)/len(y1_test):.1%})")
            
            # Before entry gate
            before_gate_signals = len(trade_indices)
            print(f"  Before Entry Gate: {before_gate_signals} potential trades")
            
            # After entry gate
            after_gate_signals = np.sum(final_preds != 0)  # Count Buy/Sell signals (0=NoTrade)
            print(f"  After Entry Gate: {after_gate_signals} confirmed trades ({after_gate_signals/len(y1_test):.1%} coverage)")
            print(f"  Entry Gate Filter: {before_gate_signals - after_gate_signals} trades filtered ({(before_gate_signals - after_gate_signals)/before_gate_signals:.1%})")
            
            # Stage 1 metrics
            stage1_acc = accuracy_score(y1_test, stage1_preds)
            print(f"  Stage 1 Accuracy: {stage1_acc:.4f}")
            
            # Stage 2 metrics
            if len(y2_trade) > 0:
                stage2_acc = accuracy_score(y2_trade, stage2_preds)
                stage2_f1 = f1_score(y2_trade, stage2_preds, average='weighted')
                print(f"  Stage 2 Accuracy: {stage2_acc:.4f}")
                print(f"  Stage 2 F1 Score: {stage2_f1:.4f}")
                
                # Overall 3-class metrics
                overall_labels = np.where(final_preds == 0, 0,  # NoTrade
                                         np.where(final_preds == 1, 1, 2))  # Sell/Buy
                true_labels = np.where(y1_test == 0, 0,  # NoTrade
                                      np.where(y2_test == 0, 1, 2))  # Sell/Buy
                
                overall_acc = accuracy_score(true_labels, overall_labels)
                overall_f1 = f1_score(true_labels, overall_labels, average='weighted')
                
                print(f"\n  Overall System:")
                print(f"    Accuracy: {overall_acc:.4f}")
                print(f"    F1 Score: {overall_f1:.4f}")
                
                # Confusion matrix
                cm = confusion_matrix(true_labels, overall_labels)
                print(f"\n  Confusion Matrix (0=NoTrade, 1=Sell, 2=Buy):")
                print(cm)
                
                # Signal distribution
                signal_counts = np.bincount(overall_labels, minlength=3)
                print(f"\n  Signal Distribution:")
                print(f"    NoTrade: {signal_counts[0]} ({signal_counts[0]/len(overall_labels):.1%})")
                print(f"    Sell: {signal_counts[1]} ({signal_counts[1]/len(overall_labels):.1%})")
                print(f"    Buy: {signal_counts[2]} ({signal_counts[2]/len(overall_labels):.1%})")
        else:
            print("  ⚠️ No Trade signals generated by Stage 1!")


def main():
    parser = argparse.ArgumentParser("NVDA 2-Stage LSTM Trading System")
    parser.add_argument("--data", "-d", default="NVDA_dss_features_20251212.csv")
    parser.add_argument("--seq_len", "-s", type=int, default=60)
    parser.add_argument("--horizon", "-H", type=int, default=5)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    args = parser.parse_args()
    
    predictor = NVDA_LSTM_2Stage(sequence_length=args.seq_len, horizon=args.horizon)
    
    print("📊 Loading and preparing data...")
    df = predictor.load_data(args.data)
    X1, y1, X2, y2 = predictor.prepare_data(df)
    
    # Create sequences
    X1_seq, y1_seq = predictor.create_sequences(X1, y1)
    X2_seq, y2_seq = predictor.create_sequences(X2, y2)
    
    # Split data
    X1_train, X1_test, y1_train, y1_test = predictor.split_data(X1_seq, y1_seq)
    X2_train, X2_test, y2_train, y2_test = predictor.split_data(X2_seq, y2_seq)
    
    # Train models
    predictor.train_stage1(X1_train, y1_train, X1_test, y1_test, epochs=args.epochs, batch_size=args.batch_size)
    predictor.train_stage2(X2_train, y2_train, X2_test, y2_test, epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate
    predictor.evaluate_3stage(X1_test, y1_test, X2_test, y2_test, df)
    
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
