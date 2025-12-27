#!/usr/bin/env python3
"""
NVDA Stock Price Prediction using LSTM with PyTorch
Regression model for price prediction and Classification model for direction prediction.

Features:
- Data preprocessing with MinMaxScaler
- Sequence generation (60-day window)
- LSTM models with dropout and early stopping
- Evaluation metrics: RMSE/MAE for regression, Accuracy/F1 for classification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class StockDataset(Dataset):
    """Custom dataset for stock data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """LSTM model for regression."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMClassifier(nn.Module):
    """LSTM model for classification."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        # Remove sigmoid as BCEWithLogitsLoss applies it internally
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class NVDA_LSTM_Predictor:
    """LSTM model for NVDA stock prediction using PyTorch."""
    
    def __init__(self, sequence_length=60):
        """
        Initialize the LSTM predictor.
        
        Args:
            sequence_length: Number of days to look back for prediction (default: 60)
        """
        self.sequence_length = sequence_length
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_columns = None
        self.target_column = 'Adj Close'
        self.model_regression = None
        self.model_classification = None
        
    def load_data(self, csv_file):
        """
        Load and prepare data from CSV file.
        
        Args:
            csv_file: Path to CSV file with features
            
        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(csv_file)
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        
        # Select feature columns (exclude Date, Index, and target-related columns to prevent data leakage)
        exclude_cols = ['Date', 'Index', 'Adj Close', 'Close', 'daily_return', 'price_change']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"✅ Loaded {len(df)} rows with {len(self.feature_columns)} features")
        print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare data for LSTM training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Scaled features and targets
        """
        # Extract features and target
        features = df[self.feature_columns].values
        target = df[self.target_column].values.reshape(-1, 1)
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(features)
        y_scaled = self.scaler_y.fit_transform(target)
        
        return X_scaled, y_scaled
    
    def create_sequences(self, X, y):
        """
        Create sequences for LSTM training.
        
        Args:
            X: Scaled features
            y: Scaled target
            
        Returns:
            X_sequences, y_sequences
        """
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def create_classification_labels(self, y):
        """
        Create classification labels (1 for price up, 0 for price down).
        
        Args:
            y: Target values (scaled, shape (N, 1))
            
        Returns:
            Binary labels (same length as input y, shape (N, 1))
        """
        # Unscale to compare actual prices
        y_unscaled = self.scaler_y.inverse_transform(y)  # (N, 1)
        
        labels = np.zeros(len(y_unscaled))
        for i in range(1, len(y_unscaled)):
            if y_unscaled[i, 0] > y_unscaled[i - 1, 0]:  # Use scalar values
                labels[i] = 1
        
        return labels.reshape(-1, 1)
    
    def split_data(self, X_seq, y_seq):
        """
        Split data into train and test sets (80/20).
        
        Args:
            X_seq: Input sequences
            y_seq: Target sequences
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Use time-based split for time series
        split_idx = int(len(X_seq) * 0.8)
        
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"📊 Data split: Train={len(X_train)} sequences, Test={len(X_test)} sequences")
        
        return X_train, X_test, y_train, y_test
    
    def train_regression_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the regression model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        print("\n🔧 Training Regression Model...")
        
        # Create datasets and dataloaders
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Build model
        input_size = X_train.shape[2]
        self.model_regression = LSTMRegressor(input_size).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_regression.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model_regression.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model_regression(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model_regression.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.model_regression(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model_regression.state_dict(), 'best_regression_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        self.model_regression.load_state_dict(torch.load('best_regression_model.pth'))
        
        print("✅ Regression model training completed!")
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def train_classification_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the classification model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        print("\n🔧 Training Classification Model...")
        
        # Create datasets and dataloaders
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Build model
        input_size = X_train.shape[2]
        self.model_classification = LSTMClassifier(input_size).to(device)
        
        # Loss and optimizer with class weights
        # Calculate class weights for imbalance
        n_up = np.sum(y_train)
        n_down = len(y_train) - n_up
        
        # Avoid division by zero
        if n_up == 0:
            pos_weight = torch.tensor([1.0], device=device)
        else:
            pos_weight = torch.tensor([n_down / n_up], device=device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model_classification.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model_classification.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.model_classification(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs.squeeze() > 0).float()  # BCEWithLogitsLoss threshold at 0
                total += batch_y.size(0)
                correct += (predicted == batch_y.squeeze()).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            self.model_classification.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.model_classification(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    predicted = (outputs.squeeze() > 0).float()  # BCEWithLogitsLoss threshold at 0
                    total += batch_y.size(0)
                    correct += (predicted == batch_y.squeeze()).sum().item()
            
            val_loss /= len(test_loader)
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model_classification.state_dict(), 'best_classification_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        self.model_classification.load_state_dict(torch.load('best_classification_model.pth'))
        
        print("✅ Classification model training completed!")
        return {'train_loss': train_losses, 'val_loss': val_losses, 'train_acc': train_accs, 'val_acc': val_accs}
    
    def evaluate_regression(self, X_test, y_test):
        """
        Evaluate regression model.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model_regression.eval()
        test_dataset = StockDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                outputs = self.model_regression(batch_X)
                predictions.extend(outputs.cpu().numpy())
        
        y_pred_scaled = np.array(predictions)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        print("\n📊 Regression Model Evaluation:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return metrics, y_true, y_pred
    
    def evaluate_classification(self, X_test, y_test):
        """
        Evaluate classification model.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model_classification.eval()
        test_dataset = StockDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = self.model_classification(batch_X)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        y_pred_prob = np.array(predictions)
        y_pred = (y_pred_prob > 0).astype(int).flatten()  # BCEWithLogitsLoss: threshold at 0
        y_true = np.array(actuals).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        metrics = {
            'Accuracy': accuracy,
            'F1 Score': f1
        }
        
        print("\n📊 Classification Model Evaluation:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return metrics, y_true, y_pred


def main():
    """Main function to run LSTM prediction."""
    parser = argparse.ArgumentParser(description="NVDA LSTM Stock Prediction with PyTorch")
    parser.add_argument("--data", "-d", default="NVDA_dss_features_20251212.csv", help="Input CSV file")
    parser.add_argument("--seq_len", "-s", type=int, default=60, help="Sequence length")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--model", "-m", choices=['regression', 'classification', 'both'], default='both', 
                       help="Model type to train")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NVDA_LSTM_Predictor(sequence_length=args.seq_len)
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    df = predictor.load_data(args.data)
    X_scaled, y_scaled = predictor.prepare_data(df)
    
    # Create sequences
    X_seq, y_seq = predictor.create_sequences(X_scaled, y_scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X_seq, y_seq)
    
    # Train regression model
    if args.model in ['regression', 'both']:
        history_reg = predictor.train_regression_model(X_train, y_train, X_test, y_test, 
                                                       epochs=args.epochs, batch_size=args.batch_size)
        
        # Evaluate
        reg_metrics, y_true, y_pred = predictor.evaluate_regression(X_test, y_test)
    
    # Train classification model
    if args.model in ['classification', 'both']:
        # Create classification labels
        y_cls = predictor.create_classification_labels(y_scaled)  # len=805
        y_cls = y_cls[predictor.sequence_length:]  # len=745 (align with sequences)
        X_seq_cls = X_seq  # len=745 (no need to slice)
        
        # Re-split for classification using binary labels
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = predictor.split_data(X_seq_cls, y_cls)
        
        history_cls = predictor.train_classification_model(X_train_cls, y_train_cls, X_test_cls, y_test_cls,
                                                          epochs=args.epochs, batch_size=args.batch_size)
        
        # Evaluate
        cls_metrics, y_true_cls, y_pred_cls = predictor.evaluate_classification(X_test_cls, y_test_cls)
    
    print("\n✅ LSTM prediction completed successfully!")


if __name__ == "__main__":
    main()
