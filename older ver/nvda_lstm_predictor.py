#!/usr/bin/env python3
"""
NVDA Stock Price Prediction using LSTM
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
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Plotting libraries not available. Plots will be skipped.")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class NVDA_LSTM_Predictor:
    """LSTM model for NVDA stock prediction."""
    
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
        self.history_regression = None
        self.history_classification = None
        
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
        
        # Select feature columns (exclude Date and Index)
        exclude_cols = ['Date', 'Index']
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
            y: Target values
            
        Returns:
            Binary labels
        """
        labels = np.zeros(len(y))
        for i in range(1, len(y)):
            if y[i] > y[i-1]:
                labels[i] = 1
        return labels[1:]  # Remove first element as it has no previous day to compare
    
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
    
    def build_regression_model(self, input_shape):
        """
        Build LSTM model for regression (price prediction).
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_classification_model(self, input_shape):
        """
        Build LSTM model for classification (direction prediction).
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_regression_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Train the regression model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        print("\n🔧 Training Regression Model...")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model_regression = self.build_regression_model(input_shape)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = self.model_regression.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        self.history_regression = history
        print("✅ Regression model training completed!")
        
        return history
    
    def train_classification_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Train the classification model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        print("\n🔧 Training Classification Model...")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model_classification = self.build_classification_model(input_shape)
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = self.model_classification.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        self.history_classification = history
        print("✅ Classification model training completed!")
        
        return history
    
    def evaluate_regression(self, X_test, y_test):
        """
        Evaluate regression model.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred_scaled = self.model_regression.predict(X_test)
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
        # Make predictions
        y_pred_prob = self.model_classification.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics = {
            'Accuracy': accuracy,
            'F1 Score': f1
        }
        
        print("\n📊 Classification Model Evaluation:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return metrics, y_test, y_pred
    
    def plot_results(self, y_true, y_pred, title="Price Prediction"):
        """
        Plot prediction results.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        if not PLOTTING_AVAILABLE:
            print(f"⚠️ Cannot plot '{title}': matplotlib not installed")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='True Price', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted Price', color='red', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_training_history(self, history, title="Training History"):
        """
        Plot training history.
        
        Args:
            history: Training history
            title: Plot title
        """
        if not PLOTTING_AVAILABLE:
            print(f"⚠️ Cannot plot '{title}': matplotlib not installed")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metrics plot
        if 'mae' in history.history:
            ax2.plot(history.history['mae'], label='Training MAE')
            ax2.plot(history.history['val_mae'], label='Validation MAE')
            metric_name = 'MAE'
        else:
            ax2.plot(history.history['accuracy'], label='Training Accuracy')
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
            metric_name = 'Accuracy'
        
        ax2.set_title(f'Model {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def save_models(self, reg_path='nvda_regression_model.h5', cls_path='nvda_classification_model.h5'):
        """
        Save trained models.
        
        Args:
            reg_path: Path to save regression model
            cls_path: Path to save classification model
        """
        if self.model_regression:
            self.model_regression.save(reg_path)
            print(f"✅ Regression model saved to {reg_path}")
        
        if self.model_classification:
            self.model_classification.save(cls_path)
            print(f"✅ Classification model saved to {cls_path}")


def main():
    """Main function to run LSTM prediction."""
    parser = argparse.ArgumentParser(description="NVDA LSTM Stock Prediction")
    parser.add_argument("--data", "-d", default="NVDA_dss_features_20251212.csv", help="Input CSV file")
    parser.add_argument("--seq_len", "-s", type=int, default=60, help="Sequence length")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--model", "-m", choices=['regression', 'classification', 'both'], default='both', 
                       help="Model type to train")
    parser.add_argument("--plot", "-p", action="store_true", help="Show plots")
    
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
        predictor.train_regression_model(X_train, y_train, X_test, y_test, 
                                        epochs=args.epochs, batch_size=args.batch_size)
        
        # Evaluate
        reg_metrics, y_true, y_pred = predictor.evaluate_regression(X_test, y_test)
        
        if args.plot:
            predictor.plot_results(y_true, y_pred, "NVDA Price Prediction - Regression")
            predictor.plot_training_history(predictor.history_regression, "Regression Training History")
    
    # Train classification model
    if args.model in ['classification', 'both']:
        # Create classification labels
        y_cls = predictor.create_classification_labels(y_scaled)
        y_seq_cls = y_seq[1:]  # Remove first element to match labels
        X_seq_cls = X_seq[1:]  # Remove first element to match labels
        
        # Re-split for classification
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = predictor.split_data(X_seq_cls, y_seq_cls)
        
        predictor.train_classification_model(X_train_cls, y_train_cls, X_test_cls, y_test_cls,
                                            epochs=args.epochs, batch_size=args.batch_size)
        
        # Evaluate
        cls_metrics, y_true_cls, y_pred_cls = predictor.evaluate_classification(X_test_cls, y_test_cls)
        
        if args.plot:
            predictor.plot_training_history(predictor.history_classification, "Classification Training History")
    
    # Save models
    predictor.save_models()
    
    print("\n✅ LSTM prediction completed successfully!")


if __name__ == "__main__":
    main()
